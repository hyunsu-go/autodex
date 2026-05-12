"""
Real-world grasp executor for xArm + Allegro hand.

Autonomous (no GUI) trajectory execution.

Execution sequence (matches RSS2026 reference: planner/inference/train/run_auto_v2.py):
    execute:  init(joint0) -> approach(traj) -> pregrasp -> grasp -> squeeze -> lift -> place
    release:  reverse_squeeze -> grasp -> pregrasp -> hand_init -> arm_return

Usage:
    executor = RealExecutor()
    executor.execute(plan_result)
    executor.release(plan_result)
    executor.shutdown()
"""
import datetime
import time
import numpy as np
from scipy.spatial.transform import Rotation

from autodex.planner import PlanResult
from autodex.utils.robot_config import (
    XARM_INIT, XARM_INSPIRE_INIT,
    ALLEGRO_INIT, ALLEGRO_LINK6_TO_WRIST,
    INSPIRE_INIT, INSPIRE_LINK6_TO_WRIST, INSPIRE_LEFT_LINK6_TO_WRIST,
)

# Per-hand config: (init_joints, link6_to_wrist, convert_fn)
def _convert_allegro(hand_pose: np.ndarray) -> np.ndarray:
    """Reorder Allegro joints: move last 4 (thumb) to front."""
    if hand_pose.ndim == 1:
        out = hand_pose.copy()
        out[:4] = hand_pose[12:]
        out[4:] = hand_pose[:12]
    else:
        out = hand_pose.copy()
        out[:, :4] = hand_pose[:, 12:]
        out[:, 4:] = hand_pose[:, :12]
    return out

def _convert_inspire(hand_pose: np.ndarray) -> np.ndarray:
    """Convert inspire qpos (radians) to controller action (0-1000).

    qpos order:   [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    action order:  [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    """
    limits = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
    if hand_pose.ndim == 1:
        q = hand_pose[:6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros(6, dtype=np.float64)
        action[0] = np.clip(action_float[5], 0, 1000)  # pinky
        action[1] = np.clip(action_float[4], 0, 1000)  # ring
        action[2] = np.clip(action_float[3], 0, 1000)  # middle
        action[3] = np.clip(action_float[2], 0, 1000)  # index
        action[4] = np.clip(action_float[1], 0, 1000)  # thumb_pitch
        action[5] = np.clip(action_float[0], 0, 1000)  # thumb_yaw
    else:
        q = hand_pose[:, :6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros_like(hand_pose)
        action[:, 0] = np.clip(action_float[:, 5], 0, 1000)
        action[:, 1] = np.clip(action_float[:, 4], 0, 1000)
        action[:, 2] = np.clip(action_float[:, 3], 0, 1000)
        action[:, 3] = np.clip(action_float[:, 2], 0, 1000)
        action[:, 4] = np.clip(action_float[:, 1], 0, 1000)
        action[:, 5] = np.clip(action_float[:, 0], 0, 1000)
    return action

HAND_CONFIG = {
    "allegro": {
        "init": ALLEGRO_INIT,
        "link6_to_wrist": ALLEGRO_LINK6_TO_WRIST,
        "convert": _convert_allegro,
        "xarm_init": XARM_INIT,
    },
    "inspire": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
    "inspire_left": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LEFT_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
}


class RealExecutor:
    def __init__(
        self,
        arm_name: str = "xarm",
        hand_name: str = "allegro",
        dt: float = 0.01,
        squeeze_level: int = 10,
    ):
        if hand_name not in HAND_CONFIG:
            raise ValueError(f"Unknown hand: {hand_name}. Choose from {list(HAND_CONFIG)}")
        self.dt = dt
        self.squeeze_level = squeeze_level
        self.hand_name = hand_name

        hcfg = HAND_CONFIG[hand_name]
        self._convert = hcfg["convert"]
        self._hand_init = hcfg["init"]
        self._link6_to_wrist = hcfg["link6_to_wrist"]
        self._xarm_init = hcfg["xarm_init"]

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm(arm_name)
        self.hand = get_hand(hand_name)

        # Safety velocity limits
        self.joint_vel_limit = 0.05
        self.cart_vel_limit = 0.002
        self.rot_vel_limit = 0.01
        self.hand_vel_limit = 0.03

    # ── low-level motion primitives ──────────────────────────────────────

    def _safe_joint_step(self, current, target, vel_limit=None):
        delta = target - current
        limit = vel_limit if vel_limit is not None else self.joint_vel_limit
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta = delta / norm * limit
        return current + delta

    def _move_joints(self, arm_traj, hand_traj=None, threshold=0.02):
        for i in range(len(arm_traj)):
            target_arm = arm_traj[i]
            target_hand = hand_traj[i] if hand_traj is not None else None
            if target_hand is not None:
                self.hand.move(target_hand)
            stall_count = 0
            prev_qpos = None
            recovered = False
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        print("[executor] stall detected, clearing error...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        print("[executor] stall after recovery, aborting")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                nxt = self._safe_joint_step(cur, target_arm)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if np.linalg.norm(self.arm.get_data()["qpos"] - target_arm) < threshold:
                    break

    def _move_hand(self, target):
        self.hand.move(target)
        time.sleep(self.dt)

    def _move_cartesian(self, target_pose, threshold_t=0.002, threshold_r=0.02,
                        vel_scale=1.0, stop_on_stall=False,
                        stall_window=30, stall_min_progress=0.001):
        """Stall detection is window-based: over the last `stall_window` ticks
        we must have advanced at least `stall_min_progress` meters (default 1 mm)
        otherwise we count as stalled. This is robust to xarm position-reading
        latency/noise on slow descents.

        stop_on_stall=True breaks immediately on stall (placing mode — stop on
        contact, don't clear_error or retry).
        """
        from collections import deque

        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        pos_history = deque(maxlen=stall_window)
        stalled = False
        recovered = False
        recover_count = 0
        for _ in range(500):
            cur = self.arm.get_data()["position"].copy()
            cur_pos = cur[:3, 3].copy()
            pos_history.append(cur_pos)
            # Stall = full window collected and total displacement is small.
            if len(pos_history) == stall_window:
                progress = np.linalg.norm(pos_history[-1] - pos_history[0])
                stalled = (progress < stall_min_progress)
            if stalled:
                if stop_on_stall:
                    print(f"[executor] stall detected (window {stall_window} ticks, "
                          f"progress {progress*1000:.2f}mm) — stopping (placing mode)")
                    break
                if not recovered:
                    print("[executor] stall detected, clearing error...")
                    self.arm.clear_error()
                    recovered = True
                    pos_history.clear()
                    stalled = False
                else:
                    recover_count += 1
                    if recover_count >= stall_window:
                        print("[executor] stall after recovery, aborting")
                        break
            prev_pos = cur_pos
            t_delta = target_pose[:3, 3] - cur[:3, 3]
            t_dist = np.linalg.norm(t_delta)
            vel = self.cart_vel_limit * vel_scale
            if t_dist > vel:
                t_delta = t_delta / t_dist * vel
            cur[:3, 3] += t_delta
            cur_rot = Rotation.from_matrix(cur[:3, :3])
            r_delta = (target_rot * cur_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)
            if r_dist > self.rot_vel_limit:
                r_delta = r_delta / r_dist * self.rot_vel_limit
            if r_dist > 0.001:
                cur[:3, :3] = (Rotation.from_rotvec(r_delta) * cur_rot).as_matrix()
            self.arm.move(cur, is_servo=True)
            time.sleep(self.dt)
            actual = self.arm.get_data()["position"]
            if (np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]) < threshold_t
                    and np.linalg.norm((target_rot * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()) < threshold_r):
                break

    def _move_joint_sequential(self, target_qpos, joint_order, threshold=0.06):
        current_target = self.arm.get_data()["qpos"].copy()
        for j in joint_order:
            current_target[j] = target_qpos[j]
            stall_count = 0
            prev_qpos = None
            recovered = False
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        print(f"[executor] joint {j} stall, clearing error...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        print(f"[executor] joint {j} stall after recovery, skipping")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                nxt = self._safe_joint_step(cur, current_target, vel_limit=0.06)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if np.abs(self.arm.get_data()["qpos"][j] - target_qpos[j]) < threshold:
                    break

    # ── public API ────────────────────────────────────────────────────────

    def start_recording(self, save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        self.arm.stop()
        self.hand.stop()

    def _log_state(self, state):
        ts = datetime.datetime.now().isoformat()
        self.state_timestamps.append({"state": state, "time": ts})

    def execute(self, plan_result: PlanResult, lift_height: float = 0.10):
        """
        Execute: init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        State timestamps stored in self.state_timestamps.
        Returns the squeezed hand pose.
        """
        if not plan_result.success:
            print("Planning failed — nothing to execute.")
            return None

        self.state_timestamps = []
        traj = plan_result.traj
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        wrist_ee = plan_result.wrist_se3 @ np.linalg.inv(self._link6_to_wrist)

        return self._execute_auto(traj, pg_hand, g_hand, wrist_ee, lift_height)

    def _execute_auto(self, traj, pg_hand, g_hand, wrist_ee, lift_height):
        """Reference: run_auto_v2.py lines 318-335"""
        sl = self.squeeze_level

        # 1. Return to init pose (joint 0 first)
        self._log_state("init")
        self._move_joint_sequential(self._xarm_init[:6], [0])

        # 2. Approach trajectory
        self._log_state("approach")
        hand_traj = np.array([self._convert(traj[i, 6:]) for i in range(len(traj))])
        self._move_joints(traj[:, :6], hand_traj)

        # 3. Pregrasp
        self._log_state("pregrasp")
        self._move_hand(pg_hand)

        # 4. Grasp
        self._log_state("grasp")
        self._move_hand(g_hand)

        # 5. Squeeze
        self._log_state("squeeze")
        for i in range(sl * 5):
            s_hand = g_hand * (1 + i / 5) - pg_hand * (i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        # 6. Lift
        self._log_state("lift")
        lift_pose = wrist_ee.copy()
        lift_pose[2, 3] += lift_height
        self._move_cartesian(lift_pose, vel_scale=1/1.5)

        # 7. Place (descend back down slowly, stop on contact).
        # Target = 5cm BELOW original grasp height — pushes into ground so that
        # perception/c2r slack still lets the object meet the table. Relies on
        # stop_on_stall to halt safely once contact is felt.
        self._log_state("place")
        place_overshoot = 0.05
        target_descend = lift_height + place_overshoot
        start_z = self.arm.get_data()["position"][2, 3]
        place_pose = self.arm.get_data()["position"].copy()
        place_pose[2, 3] -= target_descend
        self._move_cartesian(place_pose, vel_scale=1/3.0, stop_on_stall=True)
        descended = start_z - self.arm.get_data()["position"][2, 3]
        if descended < target_descend - 0.005:
            print(f"[executor] place: stopped on contact — descended {descended*1000:.1f}mm "
                  f"of target {target_descend*1000:.0f}mm")
        else:
            print(f"[executor] place: full descent (no contact!) — {descended*1000:.1f}mm "
                  f"(target {target_descend*1000:.0f}mm)")

        self._log_state("done")
        return s_hand

    def release(self, plan_result: PlanResult):
        """Release object and return arm to init pose."""
        if not plan_result.success:
            return

        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        self._release_auto(pg_hand, g_hand)

    def _release_auto(self, pg_hand, g_hand):
        """Reverse squeeze -> grasp -> pregrasp, then STOP.
        Hand opening to hand_init and arm retract back to init are intentionally
        skipped — user resets those manually after inspecting the placed object."""
        sl = self.squeeze_level

        # Reverse squeeze
        for i in range(sl * 5):
            s_hand = g_hand * (sl - i / 5) - pg_hand * (sl - 1 - i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        self._move_hand(g_hand)
        time.sleep(0.01)
        self._move_hand(pg_hand)

    def shutdown(self):
        self.arm.end()
        self.hand.end()
