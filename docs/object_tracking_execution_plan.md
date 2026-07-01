# AutoDex object tracking execution plan

작성일: 2026-06-30

이 문서는 AutoDex 코드에 이미 들어와 있는 object tracking 경로를 기준으로,
실제 실험 trial에서 `capture1`, `capture2`, `capture3`, `capture5`,
`capture6`를 사용해 GoTrack 기반 object tracking을 실행하기 위한 구현
계획을 정리한다. 기존 tracking 코드는 유지하고, orchestration, 진행률 기록,
모니터링, 업로드/재시작 안정성만 새 코드로 추가하는 방향을 전제로 한다.

작업 브랜치:

```text
tracking-session-progress
```

## 1. Git 상태와 pull 판단

`git fetch origin`으로 원격 ref는 갱신했다. 확인 시점의 상태는 다음과 같다.

```text
## main...origin/main [ahead 163, behind 71]
 M autodex/utils/symmetry.py
 M src/execution/run_auto.py
?? autodex-code
```

추가 확인:

```text
git rev-list --left-right --count main...origin/main
163  71
```

현재는 자동으로 `git pull`하지 않는 것이 맞다.

- 로컬 `main`이 `origin/main`보다 163개 앞서 있고 71개 뒤져 있다.
- 원격 `origin/main`은 fetch 과정에서 force-update된 이력이 있다.
- 작업 트리에 이미 수정된 파일 2개와 untracked 디렉터리가 있다.
- 수정된 `src/execution/run_auto.py`에는 `--no_video` 옵션과 video recording
  guard가 들어가 있어, 실험 실행 코드와 직접 관련된다.

따라서 지금 pull하면 merge/rebase 충돌뿐 아니라, object tracking 관련 로컬
작업이 예기치 않게 섞일 수 있다. 최신 GitHub 반영이 필요하면 먼저 별도
브랜치를 만들고 dirty diff를 보존한 뒤 `origin/main`과 수동 merge/rebase를
진행해야 한다.

## 2. 현재 구현되어 있는 것

### 2.1 Distributed init

이미 구현되어 있다.

- `autodex/perception/foundpose_init.py`
  - FoundPose first-frame pose init wrapper.
  - per-object `repre.pth` cache가 없으면 onboarding을 실행하는 구조.
- `autodex/perception/init_orchestrator.py`
  - robot PC에서 capture PC들의 init daemon에 `init`/`run` 명령을 보낸다.
  - mask와 pose payload를 request id 기준으로 모은다.
  - 수집 중 `masks X/N`, `poses Y/N` 형태의 live progress를 출력한다.
  - IoU pose select와 silhouette refine까지 robot PC에서 수행한다.
- `src/execution/daemon/init_daemon.py`
  - capture PC에서 실행되는 init daemon.
- `scripts/init_daemons.sh`
  - `capture1`, `capture2`, `capture3`, `capture5`, `capture6`에 SSH로 daemon을
    start/stop/status/log 관리한다.

`src/execution/run_auto.py`에는 FoundPose distributed init 경로가 이미 붙어
있다. 즉 first-frame `pose_world.npy`를 만드는 부분은 실험 loop 안에서 어느
정도 연결되어 있다.

### 2.2 Distributed GoTrack tracking

핵심 모듈은 이미 들어와 있다.

- `autodex/perception/gotrack_engine.py`
  - capture PC 한 대에서 해당 PC의 camera subset에 대해 GoTrack stage 1-4를
    실행한다.
  - raw frame을 받아 anchor observation을 만든다.
  - debug crop은 tracking loop를 막지 않도록 local `/tmp/gotrack_crops`에
    쓰고, daemon stop 후 rsync 업로드하는 구조를 갖고 있다.
- `src/execution/daemon/gotrack_daemon.py`
  - capture PC daemon.
  - `MultiCameraReader`로 SHM frame을 읽고, robot PC에서 받은 prior pose를
    사용해 `GoTrackEngine.process_frame()`을 돌린다.
  - per-camera anchor observation을 robot PC로 publish한다.
  - `init`, `start`, `stop`, `exit` control command를 받는다.
  - stop 후 debug crop을 background rsync로 NAS에 올린다.
- `autodex/perception/gotrack_tracker.py`
  - robot PC tracker.
  - capture PC별 observation을 frame id로 sync한다.
  - triangulation과 robust Kabsch fit으로 pose를 갱신한다.
  - 새 pose를 prior pose로 capture PC들에게 publish한다.
  - in-memory `status` dict에 frame id, FPS, per-PC last frame, fail reason,
    inlier/residual 통계를 누적한다.
- `autodex/dashboard/tracking_monitor.py`
  - `GoTrackTracker.status`를 Flask dashboard로 보여준다.
  - 현재 process가 살아 있는 동안의 상태를 보는 용도다.
- `scripts/gotrack_daemons.sh`
  - `capture1`, `capture2`, `capture3`, `capture5`, `capture6`에 SSH로
    GoTrack daemon을 start/stop/status/log 관리한다.

### 2.3 Offline/batch tracking 및 overlay 쪽 참고 코드

진행률과 skip/resume 패턴을 참고할 코드도 이미 있다.

- `src/process/batch_object_overlay.py`
  - episode 단위 work discovery.
  - `world_pose_records.json`이 있으면 GoTrack 완료로 간주해 skip한다.
  - local cache, prefetch, background upload 패턴을 사용한다.
  - GoTrack subprocess의 progress line을 regex로 읽어 tqdm frame progress로
    반영한다.
- `src/process/overlay_status_server.py`
  - 파일 존재 여부를 읽어 episode별 GoTrack/overlay 완료 여부를 보여준다.
- `scripts/monitor_foundpose_onboard.py`
  - process memory가 아니라 산출물 파일을 스캔해 누적 진행률을 보여준다.
  - `repre.pth` 존재 여부와 template 이미지 개수로 완료/진행률을 판단한다.
- `scripts/foundpose_nas_watcher.sh`
  - local 산출물을 감시하다가 완료 파일이 생기면 NAS로 rsync한다.
  - sync 완료 상태를 로그에 누적한다.

## 3. 현재 부족한 것

아래는 아직 production execution 관점에서 비어 있는 부분이다.

1. Live tracking session orchestration
   - FoundPose init으로 얻은 `pose_world.npy`를 사용해 GoTrack daemon들을
     `init -> start -> stop`시키고, robot PC tracker를 실행하는 상위 session
     wrapper가 없다.
   - `GoTrackTracker` 단독 CLI는 pose를 받아 track할 수 있지만, trial
     directory, daemon command, output persistence, skip/resume을 한 번에
     묶지는 않는다.

2. Durable progress
   - 현재 dashboard는 in-memory 상태만 보여준다.
   - process가 죽거나 terminal이 닫히면 누적 진행 상황을 안정적으로 복원할
     수 없다.
   - 성공 frame pose를 `world_pose_records.json`에 live로 저장하는 통합
     코드가 없다.

3. Trial output contract
   - overlay 쪽은
     `object_tracking/gotrack_output/world_pose_records.json`을 기대한다.
   - live tracking 경로에서 이 파일을 표준 형식으로 생성하고 final summary를
     남기는 wrapper가 필요하다.

4. Skip/resume policy
   - 예전 버전으로 이미 저장된 `world_pose_records.json`은 다시 만들 필요가
     없다.
   - 완료 파일이 있는 trial을 건너뛰고, partial/crashed trial은 어떤 기준으로
     재시작할지 정의해야 한다.

5. Upload state
   - debug crop upload는 daemon 내부에 있지만, trial-level tracking output과
     upload 완료 여부는 별도 상태 파일로 남지 않는다.

6. Per-capture-PC realtime status
   - 사용자는 `capture1`, `capture2`, `capture3`, `capture5`, `capture6`
     각각이 진행 중인지, stale인지, 완료됐는지 실시간으로 확인할 수 있어야
     한다.
   - 현재 `GoTrackTracker.status.per_pc_last_frame`은 각 capture PC의 마지막
     frame id와 timestamp를 갖고 있으므로 running/stale 판단의 핵심 재료는
     이미 있다.
   - 다만 daemon init 완료, start 명령 수신, stop 완료, upload 완료 같은
     lifecycle 상태는 아직 durable file로 남지 않는다.

## 4. 분배 단위 판단: episode vs object

결론부터 말하면, live object tracking과 이미 저장된 video 후처리는 분배 단위가
다르다. live object tracking은 episode/object를 capture PC에 나누어 맡기는
문제가 아니다. 한 trial의 한 frame을 추적하려면 5개 capture PC가 각자 자기
카메라 frame을 동시에 처리해야 한다. 즉 live execution의 분배 단위는
camera-owner PC이고, 작업 단위는 "현재 trial tracking session"이다.

반대로 이미 저장된 video를 offline reprocessing하는 경우에는 episode 단위
dynamic scheduling이 기본값으로 가장 적절하다.

### 4.1 Live tracking

추천:

- `capture1`, `capture2`, `capture3`, `capture5`, `capture6`는 모두 같은
  trial에 동시에 참여한다.
- 각 capture PC는 자기에게 물린 카메라 4대 정도만 처리한다.
- robot PC는 5개 PC에서 올라오는 anchor observation을 frame id 기준으로
  모아 pose를 fit한다.

이 구조에서는 "5개 PC에 episode를 나누는" 방식이 맞지 않는다. 한 episode를
추적하는 동안 5개 PC가 모두 필요하기 때문이다.

### 4.2 Offline reprocessing

이미 저장된 video episode를 나중에 batch로 다시 돌리는 상황이라면 episode
단위 queue가 낫다. 이때 episode 단위 분배가 single-view tracking을 뜻하지는
않는다. 각 worker는 자신이 claim한 episode의 모든 camera video를 함께 읽고,
episode 내부에서는 여전히 multi-view GoTrack/fusion을 수행한다.

권장 구조:

```text
shared episode queue
  ├─ capture1 worker claims obj_a/ep_001
  ├─ capture2 worker claims obj_a/ep_002
  ├─ capture3 worker claims obj_b/ep_001
  ├─ capture5 worker claims obj_c/ep_004
  └─ capture6 worker claims obj_d/ep_003

each worker:
  load all views for one episode
  run multi-view GoTrack
  write world_pose_records.json
  render overlay videos
  mark episode done/failed/skipped
```

episode 단위 장점:

- output directory가 episode별로 이미 나뉘어 있어 skip/resume이 쉽다.
- `world_pose_records.json` 존재 여부로 완료 판단이 명확하다.
- object마다 episode 수가 달라도 work stealing이 가능해 load balancing이 좋다.
- 실패한 episode만 재시도하기 쉽다.

episode 단위 단점:

- 같은 object의 mesh/anchor bank/checkpoint를 episode마다 다시 로드할 수 있어
  object 단위보다 cache 효율이 떨어질 수 있다.
- 같은 object의 episode가 여러 PC에 흩어지면 object별 디버깅 로그를 모아
  보기 불편하다.

object 단위 장점:

- object asset을 한 번 로드한 뒤 여러 episode를 연속 처리할 수 있다.
- object별 성공률/실패 원인 분석이 직관적이다.
- FoundPose onboarding처럼 object 하나의 산출물이 핵심인 작업에는 잘 맞는다.

object 단위 단점:

- object별 episode 수가 다르면 특정 PC만 오래 남는 straggler가 생긴다.
- 완료/부분완료 판단이 episode 단위보다 거칠어진다.
- 실패한 episode 하나 때문에 object 전체 작업 상태가 애매해질 수 있다.

정리하면:

- live tracking: 5개 capture PC가 한 trial에 동시에 참여한다.
- offline tracking/overlay: episode 단위 queue를 기본으로 한다.
- asset generation/onboarding: object 단위가 적합하다.

2026-07-01 기준 구현은 이 판단을 반영해 offline용 episode-level dynamic
scheduler를 별도 추가했다. live session wrapper는 유지하고, 저장된 video
후처리는 새 scheduler를 사용한다.

## 5. 새 코드 추가 계획

기존 `GoTrackEngine`, `gotrack_daemon`, `GoTrackTracker`, `run_auto.py`는
초기 구현에서는 건드리지 않는다. 아래 파일을 새로 추가하는 방향이 적절하다.

2026-06-30 현재 1차 구현은 이 원칙대로 진행했다.

- `autodex/tracking/progress.py`
- `autodex/tracking/session.py`
- `autodex/tracking/overlay_check.py`
- `src/execution/run_gotrack_session.py`
- `scripts/monitor_gotrack_progress.py`
- `scripts/gotrack_progress_dashboard.py`
- `scripts/run_gotrack_overlay_check.py`
- `autodex/tracking/episode_queue.py`
- `scripts/run_gotrack_episode_scheduler.py`
- `scripts/gotrack_episode_dashboard.py`

capture PC별 환경은 이미 구축되어 있다고 가정한다. 즉 conda env 설치,
GoTrack checkpoint 배포, anchor bank 생성/동기화, AutoDex 코드 동기화는 이
wrapper의 책임이 아니다. 이 구현은 이미 떠 있는 `gotrack_daemon`에 command를
보내고, robot PC tracker output을 trial directory에 안정적으로 기록하는 데
집중한다.

offline 후처리에서는 `gotrack_daemon`을 사용하지 않는다. 대신
`scripts/run_gotrack_episode_scheduler.py`가 shared schedule directory를 만들고,
각 capture PC worker가 episode를 하나씩 claim한 뒤 기존
`src/process/batch_object_overlay.py`를 episode 단위로 실행한다. 이 기존 batch
worker는 episode 안의 모든 view를 읽어 multi-view GoTrack과 overlay를 수행한다.

### 5.1 `autodex/tracking/progress.py`

역할:

- append-only `events.jsonl` writer.
- atomic `state.json` writer.
- final `summary.json` writer.
- `world_pose_records.jsonl` live writer와 final
  `world_pose_records.json` 변환 helper.

출력 위치:

```text
<trial_dir>/object_tracking/gotrack_output/
  run_manifest.json
  events.jsonl
  state.json
  world_pose_records.jsonl
  world_pose_records.json
  summary.json
  logs/
```

`events.jsonl` event 예시:

```json
{"ts": 1782800000.1, "event": "session_created", "obj": "pepsi_light", "trial": "20260630_120000"}
{"ts": 1782800001.2, "event": "preflight_done", "expected_pcs": ["capture1", "capture2", "capture3", "capture5", "capture6"]}
{"ts": 1782800002.0, "event": "daemon_init_sent", "pc": "capture1"}
{"ts": 1782800005.4, "event": "daemon_start_sent"}
{"ts": 1782800006.1, "event": "first_observation", "pc": "capture2", "frame_id": 101}
{"ts": 1782800006.3, "event": "pose_written", "frame_id": 101, "n_inliers": 183, "mean_residual_mm": 4.2}
{"ts": 1782800030.0, "event": "session_done", "frames_success": 420, "frames_failed": 7}
```

`state.json` 예시:

```json
{
  "phase": "tracking",
  "obj": "pepsi_light",
  "trial_dir": "...",
  "started_at": 1782800000.1,
  "updated_at": 1782800006.3,
  "expected_pcs": ["capture1", "capture2", "capture3", "capture5", "capture6"],
  "connected_pcs": ["capture1", "capture2", "capture3", "capture5"],
  "last_frame_id": 101,
  "fps": 8.7,
  "frames_received": 108,
  "frames_success": 101,
  "frames_failed": 7,
  "fail_by_reason": {"triangulation_empty": 5, "fit_failed": 2},
  "per_pc_last_frame": {
    "capture1": {"frame_id": 101, "age_s": 0.04},
    "capture2": {"frame_id": 101, "age_s": 0.03}
  },
  "output_files": {
    "events": "events.jsonl",
    "state": "state.json",
    "poses_live": "world_pose_records.jsonl",
    "poses_final": "world_pose_records.json"
  },
  "upload": {"status": "not_started"}
}
```

별도로 `capture_pc_status.json`도 atomic write한다. 이 파일은 5개 PC를
고정 key로 갖고, monitor가 한눈에 per-PC 상태를 보여주도록 한다.

```json
{
  "updated_at": 1782800006.3,
  "pcs": {
    "capture1": {
      "ip": "192.168.0.101",
      "phase": "running",
      "daemon_seen": true,
      "init_sent": true,
      "start_sent": true,
      "stop_sent": false,
      "first_obs_ts": 1782800005.9,
      "last_obs_ts": 1782800006.3,
      "last_frame_id": 101,
      "last_obs_age_s": 0.04,
      "frames_received": 101,
      "engine_sec_ema": 0.16,
      "status": "running"
    },
    "capture2": {
      "ip": "192.168.0.102",
      "phase": "running",
      "daemon_seen": true,
      "init_sent": true,
      "start_sent": true,
      "stop_sent": false,
      "first_obs_ts": 1782800006.0,
      "last_obs_ts": 1782800004.6,
      "last_frame_id": 97,
      "last_obs_age_s": 1.7,
      "frames_received": 97,
      "engine_sec_ema": 0.22,
      "status": "stale"
    }
  }
}
```

status 판정 규칙:

- `not_started`: session 생성 전 또는 preflight 전.
- `daemon_missing`: preflight에서 daemon process 확인 실패.
- `init_sent`: daemon에 `init` command를 보낸 상태.
- `start_sent`: daemon에 `start` command를 보냈지만 아직 observation을 못 받은
  상태.
- `running`: 최근 observation age가 threshold 이하인 상태.
- `stale`: observation을 받은 적은 있지만 최근 age가 threshold를 넘긴 상태.
- `stopping`: `stop` command를 보낸 상태.
- `complete`: session finalization이 끝났고 해당 PC가 stop 대상에 포함된 상태.
- `failed`: command 전송 실패, daemon missing, 장시간 stale 등으로 실패 처리한
  상태.

초기 구현에서는 기존 `gotrack_daemon.py`를 수정하지 않고 robot-side에서
관측 가능한 정보만 사용한다. 즉 `running/stale`은 anchor observation 수신으로
판정하고, `init/start/stop/complete`는 orchestrator가 보낸 command와 session
finalize 상태로 판정한다. init 완료를 capture PC 내부에서 정확히 ack해야 하는
요구가 생기면, 그때 별도 새 status sidecar 또는 새 `status` command를 추가하는
2단계 작업으로 분리한다.

### 5.2 `autodex/tracking/session.py`

역할:

- trial directory와 object 이름을 받아 live GoTrack tracking session을
  구성한다.
- camera calibration, mesh path, anchor bank path, init pose를 검증한다.
- `CommandSender(pc_list, port=6892)`로 gotrack daemon에 `init`, `start`,
  `stop`을 보낸다.
- `GoTrackTracker`를 생성하고 `track(init_pose_world)` generator를 소비한다.
- 성공 frame pose를 `world_pose_records.jsonl`에 즉시 append한다.
- 일정 주기로 `tracker.status`를 읽어 `state.json`을 갱신한다.
- `tracker.status.per_pc_last_frame`과 command 전송 기록을 합쳐
  `capture_pc_status.json`을 갱신한다.
- 종료 시 `world_pose_records.json`과 `summary.json`을 만든다.

daemon init payload는 기존 `gotrack_daemon.py`가 기대하는 형식을 그대로
사용한다.

필수 입력:

- `--trial-dir`
- `--obj-name`
- `--mesh-path`
- `--anchor-bank-path`
- `--init-pose-npy`
- `--cam-param-dir`
- `--pc-list capture1 capture2 capture3 capture5 capture6`
- `--capture-ips ...`

capture PC별 conda environment, GoTrack checkpoint, anchor bank, AutoDex 코드,
카메라 SHM publisher는 이미 구축되어 있다고 가정한다. 이 wrapper는 설치나
asset rsync를 수행하지 않고, 기존 daemon에 command를 보내고 진행 상태를
기록하는 역할만 맡는다.

기본 skip 기준:

- `object_tracking/gotrack_output/world_pose_records.json`이 존재하고,
  하나 이상의 record에 `pose_world`가 있으면 완료로 간주한다.
- `--force`가 없으면 완료 trial은 건너뛴다.
- `state.json`이 `phase=failed` 또는 final JSON이 없고 JSONL만 있으면
  partial로 보고 재실행 또는 recover 정책을 선택한다. 초기 구현은 안전하게
  재실행하되 기존 partial은 `archive/`로 이동하지 않고 그대로 둔다.

### 5.3 `src/execution/run_gotrack_session.py`

역할:

- standalone CLI entrypoint.
- `run_auto.py`를 수정하지 않고도 저장된 trial 또는 현재 trial에 대해 tracking
  session을 실행할 수 있게 한다.

예상 사용:

```bash
python src/execution/run_gotrack_session.py \
  --trial-dir ~/shared_data/AutoDex/experiment/<exp>/<scene>/<hand>/<obj>/<episode> \
  --obj-name <obj> \
  --pc-list capture1 capture2 capture3 capture5 capture6 \
  --capture-ips <ip1> <ip2> <ip3> <ip5> <ip6> \
  --max-frames 1000
```

진행률 확인:

```bash
python scripts/monitor_gotrack_progress.py \
  --trial-dir ~/shared_data/AutoDex/experiment/<exp>/<scene>/<hand>/<obj>/<episode> \
  --watch
```

브라우저 창에서 확인:

```bash
python scripts/gotrack_progress_dashboard.py \
  --trial-dir ~/shared_data/AutoDex/experiment/<exp>/<scene>/<hand>/<obj>/<episode> \
  --host 0.0.0.0 \
  --port 8766 \
  --open-browser
```

로봇 PC가 headless이거나 원격 접속 중이면 `--open-browser`를 빼고 출력된 URL을
로컬 브라우저에서 열면 된다. 같은 네트워크의 다른 장비에서 보려면
`--host 0.0.0.0`로 띄운 뒤 `http://<robot-pc-ip>:8766/`에 접속한다.

SSH tunnel로 안전하게 보는 방법:

```bash
# local laptop에서 실행
ssh -L 8766:127.0.0.1:8766 robot@<robot-pc-ip>

# robot PC shell에서 실행
python scripts/gotrack_progress_dashboard.py \
  --trial-dir <trial_dir> \
  --host 127.0.0.1 \
  --port 8766
```

그 다음 local browser에서 아래 주소를 연다.

```text
http://127.0.0.1:8766/
```

이 file-based dashboard가 권장 실시간 창이다. `run_gotrack_session.py`의
`--web-port`는 `GoTrackTracker` process 내부 상태를 보여주는 보조 dashboard라서
process가 죽으면 상태가 사라진다. 반면 `scripts/gotrack_progress_dashboard.py`는
`state.json`, `capture_pc_status.json`, `runs_latest.json`, `events.jsonl`을
읽기 때문에 run 종료 후에도 완료 상태와 overlay playback을 계속 볼 수 있다.

ETA 표시 규칙:

- `--max-frames N`: 현재 성공 pose 수와 tracker FPS를 기준으로 남은 frame과
  ETA를 계산한다.
- `--max-seconds S`: 시작 시각과 목표 runtime 기준으로 남은 시간을 계산한다.
- 둘 다 지정된 경우 먼저 도달할 조건을 기준으로 ETA를 표시한다.
- 둘 다 없으면 live tracking은 외부 stop 또는 실험 종료 이벤트에 의해 끝나므로
  남은 시간은 `open-ended`로 표시한다.

daemon과 tracking runtime을 건드리지 않는 입력 검증 dry-run:

```bash
python src/execution/run_gotrack_session.py \
  --trial-dir <trial_dir> \
  --obj-name <obj> \
  --capture-ips <ip1> <ip2> <ip3> <ip5> <ip6> \
  --dry-run \
  --skip-daemon-check
```

tracking 직후 overlay 확인까지 같이 실행:

```bash
python src/execution/run_gotrack_session.py \
  --trial-dir <trial_dir> \
  --obj-name <obj> \
  --capture-ips <ip1> <ip2> <ip3> <ip5> <ip6> \
  --run-overlay-check
```

이미 완료된 tracking 결과에 대해 overlay만 나중에 실행:

```bash
python scripts/run_gotrack_overlay_check.py \
  --trial-dir <trial_dir> \
  --obj-name <obj>
```

초기에는 standalone으로 검증하고, 안정화 후 `run_auto.py`에 최소 hook을
붙일지 결정한다. 사용자가 요청한 "기존 코드는 그대로" 원칙 때문에 첫 단계에서
`run_auto.py`를 수정하지 않는다.

### 5.4 `scripts/monitor_gotrack_progress.py`

역할:

- `scripts/monitor_foundpose_onboard.py`처럼 파일 기반으로 누적 진행률을 보여준다.
- live process가 죽어도 `state.json`, `events.jsonl`, `world_pose_records.jsonl`
  만으로 마지막 상태를 확인할 수 있다.

예상 출력:

```text
GoTrack tracking  14:22:10
pepsi_light/20260630_120000  [tracking]  frames 420 ok / 7 fail  fps 8.7
capture1  running   frame 420  age 0.03s  frames 420  engine 0.15s
capture2  running   frame 420  age 0.05s  frames 420  engine 0.17s
capture3  running   frame 419  age 0.18s  frames 419  engine 0.18s
capture5  running   frame 420  age 0.04s  frames 420  engine 0.16s
capture6  stale     frame 417  age 1.42s  frames 417  engine 0.24s
fail: triangulation_empty=5 fit_failed=2
output: .../object_tracking/gotrack_output/world_pose_records.jsonl
```

session 완료 후에는 다음처럼 5개 PC 각각의 완료 상태를 보여준다.

```text
GoTrack tracking  14:31:02
pepsi_light/20260630_120000  [done]  frames 612 ok / 11 fail  duration 74.2s
capture1  complete  frames 612  last frame 612
capture2  complete  frames 612  last frame 612
capture3  complete  frames 611  last frame 611
capture5  complete  frames 612  last frame 612
capture6  complete  frames 608  last frame 608
output: .../object_tracking/gotrack_output/world_pose_records.json
```

### 5.5 Optional file-based status server

`autodex/dashboard/tracking_monitor.py`는 현재 process-local dashboard다. 별도로
추가한 `scripts/gotrack_progress_dashboard.py`는 file-based dashboard다. 이
대시보드는 tracking process 메모리를 보지 않고 아래 파일만 읽는다.

- `state.json`
- `capture_pc_status.json`
- `summary.json`
- `run_manifest.json`
- `events.jsonl`

따라서 tracking process가 종료되어도 마지막 상태를 브라우저에서 확인할 수
있다. 화면은 1초마다 `/api/status`를 polling한다.

대시보드 패널별 책임은 중복되지 않도록 아래처럼 나눈다.

- `Current Run Snapshot`: 지금 선택한 run 하나의 대표 상태만 보여준다. object,
  episode/trial, phase, pose record 수, tracker FPS, fit success ratio가 여기에
  들어간다.
- `Runtime Forecast`: 현재 run의 elapsed time, 남은 ETA, 목표 대비 진행률,
  throughput을 보여준다. `--max-frames` 또는 `--max-seconds`가 지정된 run은
  목표 기반 ETA를 계산하고, 둘 다 없는 live run은 외부 stop 대기 상태이므로
  `open-ended`로 표시한다.
- `Lifecycle`: preflight, daemon init, tracking, overlay, complete 순서 중 현재
  run이 어디까지 진행됐는지 보여준다. 이것은 stage 관점이고, frame/PC 세부
  상태와 분리한다.
- `Distributed Capture Health`: `capture1`, `capture2`, `capture3`, `capture5`,
  `capture6` 각각의 live health를 보여준다. daemon 상태, 명령 결과, 마지막
  frame, frame age, frame share를 확인하는 곳이다.
- `Tracking Quality Signals`: pose fit 품질만 보여준다. last fit, residual,
  failure reason count, pose output final/live 여부를 확인한다.
- `Accumulated Run History`: 전역 `runs_latest.json`에 쌓인 object/episode별
  누적 목록이다. 완료된 run도 남고, overlay 영상 버튼도 여기에서 연다.
- `Current Run Event Log`: 현재 선택한 run의 `events.jsonl` tail이다. session
  lifecycle과 progress event를 시간순으로 디버깅할 때 사용한다.
- `Overlay Playback`: history의 카메라 serial 버튼을 눌렀을 때 같은 화면에서
  열리는 영상 player다. tracking 결과 검증용이며 다른 패널과 상태 정보를
  중복 표시하지 않는다.

Run History의 `Overlay` 컬럼은 단순 상태 표시가 아니라, 각 run의
`overlay_output_dir` 또는 기본 경로인
`<trial_dir>/object_tracking/overlay_check/`를 스캔해 `overlay_*.mp4`를
카메라 serial별 재생 버튼으로 노출한다. 버튼을 클릭하면 같은 대시보드 화면
안에 video player가 열리고, 대시보드 서버의 `/overlay?path=...` endpoint가
해당 MP4를 stream한다. 임의 파일을 열지 않도록 이 endpoint는 dashboard가 현재
run history에서 발견한 `overlay_*.mp4`만 허용한다.

추가로 더 체계적인 시각화를 위해 1차로 `Runtime Forecast`, `Lifecycle` strip,
capture PC별 `Frame Share` bar를 넣었다. 이후 실제 run 로그가 충분히 쌓이면
다음 요소를 추가할 수 있다.

- PC별 frame lag timeline: 어느 capture PC가 언제부터 늦어졌는지 확인.
- residual trend chart: pose fit residual이 시간에 따라 drift하는지 확인.
- failure reason stacked bar: 실패 원인이 카메라 부족, triangulation 실패,
  Kabsch 실패 중 어디에 몰리는지 확인.
- overlay thumbnail strip: 카메라별 overlay MP4의 대표 frame을 한눈에 비교.
- run queue view: 완료, 진행 중, skip, failed episode를 batch 관점에서 관리.

### 5.6 Upload watcher

tracking output은 trial directory 아래에 바로 쓰는 것이 기본이다. local cache를
쓰는 offline/batch 모드에서는 `batch_object_overlay.py`와
`foundpose_nas_watcher.sh` 패턴을 따라 background upload 상태를 `state.json`에
남긴다.

원칙:

- latency-sensitive tracking loop에서는 local append만 한다.
- heavy copy/rsync는 stop 후 background thread/process로 수행한다.
- upload 시작/성공/실패를 `events.jsonl`와 `state.json.upload`에 기록한다.

### 5.7 Offline episode-level dynamic scheduler

이미 저장된 video 후처리는 아래 새 파일을 사용한다.

- `autodex/tracking/episode_queue.py`
  - shared filesystem 기반 queue/status store.
  - task 단위는 `<hand>/<obj>/<episode>`.
  - task claim은 `<schedule_dir>/claims/<task_id>.lock` directory를
    `os.mkdir`로 만드는 방식이라 여러 worker가 동시에 접근해도 한 worker만
    claim한다.
  - task 상태는 `<schedule_dir>/tasks/<task_id>.json`에 저장된다.
  - worker 상태는 `<schedule_dir>/workers/<worker_id>.json`에 저장된다.
  - event log는 `<schedule_dir>/events.jsonl`에 append된다.
- `scripts/run_gotrack_episode_scheduler.py`
  - `--mode init`: 저장된 episode를 discover해 queue 생성.
  - `--mode worker`: 한 PC에서 queue를 계속 claim하며 episode 처리.
  - `--mode launch`: `capture1`, `capture2`, `capture3`, `capture5`,
    `capture6`에 ssh로 worker 실행.
  - `--mode status`: terminal에서 queue 요약 확인.
- `scripts/run_batch_object_overlay_with_env.py`
  - 기존 `src/process/batch_object_overlay.py`는 수정하지 않고 그대로 둔다.
  - worker가 실제 batch script를 실행하기 전에 PC별 conda path 차이를
    흡수한다.
  - GoTrack Python은 `AUTODEX_GOTRACK_PY`/`GOTRACK_PY` override를 먼저 보고,
    기본 후보로 `~/anaconda3/envs/gotrack_cu128/bin/python`을 사용한다.
  - overlay Python은 `AUTODEX_FPOSE_PY`/`FPOSE_PY` override를 먼저 보고,
    기본 후보로 `~/anaconda3/envs/planner/bin/python`을 사용한다.
  - 선택된 env의 `nvidia/nccl/lib`를 `LD_LIBRARY_PATH`에 추가해 overlay
    rendering의 `torch`/`nvdiffrast` 로딩 실패를 줄인다.
- `scripts/gotrack_episode_dashboard.py`
  - offline queue 전용 browser dashboard.
  - 전역 queue 상태, PC별 현재 episode ETA, 병목/실패 신호, episode별
    산출물, scheduler event log, overlay video player를 분리해서 보여준다.

기존 `src/process/batch_object_overlay.py`는 그대로 재사용한다. scheduler worker는
claim한 episode에 대해 wrapper를 거쳐 아래와 같은 단일 episode command를
실행한다.

```bash
python scripts/run_batch_object_overlay_with_env.py \
  --hand <hand> \
  --obj <obj> \
  --ep <episode>
```

이 command는 episode 안의 모든 camera video를 함께 사용해 GoTrack
`world_pose_records.json`을 만들고, 이어서 camera serial별
`overlay_<serial>.mp4`를 생성한다. 즉 분배 단위는 episode지만 tracking 자체는
multi-view다.

offline scheduler dashboard는 live dashboard와 구성이 다르다. episode-level
dynamic scheduling에서는 아직 실행하지 않은 episode가 특정 PC에 미리 배정되어
있지 않다. 따라서 PC별 ETA는 "그 PC가 현재 claim해서 처리 중인 episode의
남은 시간"을 뜻하고, 전체 queue의 남은 시간은 별도의 global ETA로 표시한다.

- `Queue Overview`: 전체 schedule의 완료율, 완료/진행/대기/실패 episode 수,
  active PC 수, global ETA, episode/hour throughput, 생성된 overlay video 수.
  전역 상태만 보여주며 PC별 세부 정보는 포함하지 않는다.
- `PC ETA`: `capture1`, `capture2`, `capture3`, `capture5`, `capture6`를
  한 줄씩 보여준다. 각 줄에는 worker state, 현재 object/episode, stage,
  frame progress, current task ETA, 완료/실패 수, 평균 episode runtime,
  마지막 업데이트 시간이 표시된다. worker 파일이 아직 없으면 `not_started`로
  남는다.
- `Bottlenecks`: 실패 episode, stale worker, 가장 오래 실행 중인 task,
  완료되었지만 overlay 산출물이 부족한 episode를 따로 모아 보여준다. 이는
  queue 요약이 아니라 재시도/점검이 필요한 항목을 찾기 위한 패널이다.
- `Episode Work Queue`: episode별 상태, object, episode id, stage,
  frame progress, worker, task ETA, runtime, output path를 보여준다. 완료된
  episode에 `overlay_*.mp4`가 있으면 버튼으로 노출하고, 클릭하면 같은 화면의
  video player에서 바로 재생한다.
- `Scheduler Events`: `events.jsonl` tail이다. queue claim, task start/done/fail,
  worker start/stop 같은 순차 이벤트만 보여주며 현재 상태 판단은 위 패널들이
  담당한다.

이 구성은 live dashboard의 `Distributed Capture Health`와 다르다. offline에서는
5개 PC가 같은 episode의 view를 나누어 처리하는 것이 아니라 서로 다른 episode를
claim하므로, PC별 카메라 health보다 worker별 episode throughput과 queue ETA가
핵심이다.

2026-07-01 실제 단일 episode 검증:

- episode:
  `/home/robot/shared_data/AutoDex/experiment/selected_100/allegro/attached_container/20260330_164351`
- GoTrack stage:
  - 24개 camera video를 함께 사용했다.
  - `total_frames=268`, `world_pose_records.json`의 pose record 268개 생성.
  - 산출물:
    `<episode>/object_tracking/gotrack_output/world_pose_records.json`
- overlay stage:
  - `frame_done=268`, `frame_total=268`, `progress_ratio=1.0`으로
    scheduler task JSON에 누적 확인.
  - camera별 `overlay_*.mp4` 24개 생성.
  - 산출물:
    `/home/robot/shared_data/AutoDex/object_overlay_video/allegro/attached_container/20260330_164351/`
- dashboard:
  - schedule API에서 `counts={"done": 1}`, `done_like=1`, `n_tasks=1` 확인.
  - `/overlay?path=...` endpoint가 실제 MP4에 대해 Range request `206`과
    `Content-Type: video/mp4`를 반환함을 확인.
  - 예시 화면 이미지:
    `docs/gotrack_episode_dashboard_actual_execution.png`

## 6. 결과 경로

중요: 코드 repo는 `/home/robot/AutoDex`지만, 실험 결과 기본 저장 위치는
`autodex/utils/path.py`의 `project_dir`인 `/home/robot/shared_data/AutoDex`다.

### 6.1 Episode/trial의 의미

이 문서에서 `episode`는 object tracking이 처리하는 가장 작은 실험 단위다.
AutoDex 코드와 결과 경로에서는 보통 `trial` 또는 timestamp directory
`<dir_idx>`로도 나타난다.

정확히는 다음을 뜻한다.

- 하나의 object, hand, scene 조합에 대해 한 번 실행되어 저장된 연속 기록.
- 하나의 timestamp directory, 예를 들어 `20260630_153012`.
- 여러 frame으로 이루어진 시간 구간.
- 여러 camera serial의 동기화된 video와 calibration을 함께 가진 단위.
- FoundPose init pose, GoTrack pose record, overlay check 결과가 같은 directory
  아래에 붙는 단위.

즉 episode는 object도 아니고, frame도 아니고, camera도 아니고, capture PC도
아니다. 예를 들어 `pepsi_light`라는 object에 대해 grasp/execution을 여러 번
반복하면 episode가 여러 개 생긴다.

```text
object:  pepsi_light
episode: 20260630_153012
frames:  0..612
cameras: 24080331, 25305460, ...
PCs:     capture1, capture2, capture3, capture5, capture6
```

live tracking에서는 하나의 episode를 처리할 때 5개 capture PC가 모두 동시에
참여한다. 따라서 live mode에서 episode를 5개 PC에 나누어 배정하지 않는다.
각 PC는 같은 episode의 자기 camera subset을 처리한다.

offline/batch reprocessing에서는 episode가 queue item이 된다. 이미 저장된
video를 다시 tracking/overlay할 때는 episode별로 완료 여부가 명확하고,
`world_pose_records.json`과 `overlay_status.json`으로 skip/resume 판단을 하기
쉽기 때문이다.

`run_auto.py`가 만드는 trial directory는 다음 규칙을 따른다.

```text
/home/robot/shared_data/AutoDex/experiment/<exp_name>/<sub>/<obj>/<dir_idx>
```

여기서:

- `<exp_name>`: `--exp_name`; 지정하지 않으면 `--grasp_version`
- `<dir_idx>`: `YYYYmmdd_HHMMSS`
- `<sub>`:
  - `scene=table`: `<hand>`
  - `scene=wall/shelf/cluttered`: `<scene>/<hand>`
  - `--success_only`가 붙으면 scene prefix 뒤에 `_success_only`가 붙는다.

예시:

```text
# table scene
/home/robot/shared_data/AutoDex/experiment/v7/inspire_left/pepsi_light/20260630_153012

# shelf scene
/home/robot/shared_data/AutoDex/experiment/v7/shelf/inspire_left/pepsi_light/20260630_153012

# shelf + success_only
/home/robot/shared_data/AutoDex/experiment/v7/shelf_success_only/inspire_left/pepsi_light/20260630_153012
```

object tracking 결과는 위 trial directory 아래에 고정적으로 저장된다.

```text
<trial_dir>/object_tracking/gotrack_output/
```

따라서 실제 파일 경로는 다음과 같다.

```text
<trial_dir>/object_tracking/gotrack_output/run_manifest.json
<trial_dir>/object_tracking/gotrack_output/events.jsonl
<trial_dir>/object_tracking/gotrack_output/state.json
<trial_dir>/object_tracking/gotrack_output/capture_pc_status.json
<trial_dir>/object_tracking/gotrack_output/world_pose_records.jsonl
<trial_dir>/object_tracking/gotrack_output/world_pose_records.json
<trial_dir>/object_tracking/gotrack_output/summary.json
<trial_dir>/object_tracking/gotrack_output/logs/
```

overlay 확인 결과는 별도 디렉터리에 저장된다.

```text
<trial_dir>/object_tracking/overlay_check/
```

실제 파일 경로:

```text
<trial_dir>/object_tracking/overlay_check/overlay_status.json
<trial_dir>/object_tracking/overlay_check/overlay_check.log
<trial_dir>/object_tracking/overlay_check/overlay_<camera_serial>.mp4
<trial_dir>/object_tracking/overlay_check/_cam_param_overlay/
```

완료/진행/실패 run 목록은 trial 밖의 전역 인덱스에 누적된다.

```text
/home/robot/shared_data/AutoDex/object_tracking/gotrack_runs/runs.jsonl
/home/robot/shared_data/AutoDex/object_tracking/gotrack_runs/runs_latest.json
```

- `runs.jsonl`: append-only history. run 상태 변화가 계속 쌓인다.
- `runs_latest.json`: run_id별 마지막 상태. dashboard의 Run History 표가 읽는 파일.
- 기본 run_id: `<obj>_<trial_name>_<trial_dir_sha1_12>`
- overlay check를 실행한 run은 `overlay_status`, `overlay_output_dir`,
  `overlay_n_outputs`, `overlay_files`가 history에 누적된다. 예전 기록에
  `overlay_files`가 없더라도 dashboard는 `overlay_output_dir` 또는 기본
  overlay 디렉터리에서 MP4를 다시 찾아 링크를 만든다.

각 파일의 용도:

- `run_manifest.json`: 실행 입력, object, trial, PC list, port, mesh/anchor/calib 경로.
- `events.jsonl`: append-only event log. session lifecycle과 frame progress 기록.
- `state.json`: 전체 session 상태. phase, FPS, frame count, fail reason 집계.
- `capture_pc_status.json`: 5개 capture PC별 상태. dashboard/monitor의 핵심 입력.
- `world_pose_records.jsonl`: tracking 중 frame pose를 즉시 append하는 live pose log.
- `world_pose_records.json`: 종료 시 생성되는 최종 pose record. 기존 overlay 코드가 읽는 파일.
- `summary.json`: 종료 요약. 성공 frame 수, runtime, residual 평균, PC별 최종 상태.
- `logs/`: 추후 stdout/stderr 또는 sidecar log를 넣기 위한 예약 디렉터리.
- `overlay_status.json`: overlay check 상태. 입력 video/record/cam_param, command, 생성된 mp4 목록.
- `overlay_<camera_serial>.mp4`: 카메라별 mesh overlay 확인 영상.

offline episode scheduler 상태는 별도 schedule directory에 저장된다.

```text
/home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/<schedule_id>/
```

구체적인 파일:

```text
<schedule_dir>/manifest.json
<schedule_dir>/events.jsonl
<schedule_dir>/tasks/<hand>__<obj>__<episode>.json
<schedule_dir>/claims/<hand>__<obj>__<episode>.lock/claim.json
<schedule_dir>/workers/<worker_id>.json
<schedule_dir>/logs/<task_id>.<worker_id>.log
<schedule_dir>/launcher_logs/<worker_id>.log
```

offline overlay 결과는 기존 batch 코드 규약에 따라 아래에 저장된다.

```text
/home/robot/shared_data/AutoDex/object_overlay_video/<hand>/<obj>/<episode>/overlay_<camera_serial>.mp4
```

예를 들어 table scene에서 `exp_name=v7`, `hand=inspire_left`,
`obj=pepsi_light`, `dir_idx=20260630_153012`라면 최종 overlay 입력 파일은
아래다.

```text
/home/robot/shared_data/AutoDex/experiment/v7/inspire_left/pepsi_light/20260630_153012/object_tracking/gotrack_output/world_pose_records.json
```

5개 PC별 실시간 상태 파일은 아래다.

```text
/home/robot/shared_data/AutoDex/experiment/v7/inspire_left/pepsi_light/20260630_153012/object_tracking/gotrack_output/capture_pc_status.json
```

대시보드 실행 예:

```bash
python scripts/gotrack_progress_dashboard.py \
  --trial-dir /home/robot/shared_data/AutoDex/experiment/v7/inspire_left/pepsi_light/20260630_153012 \
  --host 0.0.0.0 \
  --port 8766
```

이때 브라우저 URL은:

```text
http://<robot-pc-ip>:8766/
```

완료된 run의 overlay 영상은 대시보드 하단 `Run History` 표의 `Overlay`
컬럼에서 확인한다. 예를 들어 `overlay_24080331.mp4`가 생성되어 있으면
해당 카메라 serial이 버튼으로 표시되고, 클릭 시 같은 화면의 player에서
재생된다.

터미널 monitor 실행 예:

```bash
python scripts/monitor_gotrack_progress.py \
  --trial-dir /home/robot/shared_data/AutoDex/experiment/v7/inspire_left/pepsi_light/20260630_153012 \
  --watch
```

overlay check 실행 예:

```bash
python scripts/run_gotrack_overlay_check.py \
  --trial-dir /home/robot/shared_data/AutoDex/experiment/v7/inspire_left/pepsi_light/20260630_153012 \
  --obj-name pepsi_light
```

offline episode scheduler 실행 예:

```bash
# 1. episode queue 생성
python scripts/run_gotrack_episode_scheduler.py \
  --mode init \
  --hand inspire \
  --obj pepsi_light tissue_box \
  --schedule-id gotrack_inspire_20260701 \
  --stages both

# 2. 5개 capture PC에 worker launch
python scripts/run_gotrack_episode_scheduler.py \
  --mode launch \
  --schedule-id gotrack_inspire_20260701 \
  --pcs capture1 capture2 capture3 capture5 capture6 \
  --repo-dir /home/robot/AutoDex

# 3. terminal 상태 확인
python scripts/run_gotrack_episode_scheduler.py \
  --mode status \
  --schedule-id gotrack_inspire_20260701

# 4. offline queue dashboard
python scripts/gotrack_episode_dashboard.py \
  --schedule-dir /home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_inspire_20260701 \
  --host 0.0.0.0 \
  --port 8767
```

local에서 먼저 scheduler 동작만 검증하고 싶으면:

```bash
python scripts/run_gotrack_episode_scheduler.py \
  --mode worker \
  --schedule-id gotrack_inspire_20260701 \
  --worker-id local-test \
  --once \
  --dry-run
```

기본 video 탐색 순서:

```text
<trial_dir>/videos
<trial_dir>/raw/exec/videos
<trial_dir>/raw/exec
<trial_dir>/raw/place/videos
<trial_dir>/raw/place
<trial_dir>/raw/videos
<trial_dir>/raw
```

주의: 현재 overlay check 1차 구현은 `world_pose_records.json`의 `frame_index`가
video frame index와 직접 대응된다고 가정한다. 기존 offline GoTrack/overlay
경로는 이 가정과 맞는다. live execution에서 hardware frame id와 AVI frame index가
어긋나는 경우에는 `raw/timestamps/frame_id.npy` 또는 `timestamp.npy`를 이용해
record를 remap하는 2차 보강이 필요하다.

## 7. 실행 흐름

### 7.1 Preflight

1. `scripts/gotrack_daemons.sh status`로 5개 PC daemon 상태 확인.
2. trial directory에 다음 파일이 있는지 확인.
   - `pose_world.npy`
   - `cam_param/intrinsics.json`
   - `cam_param/extrinsics.json`
3. object asset 확인.
   - mesh: `~/shared_data/AutoDex/object/paradex/<obj>/raw_mesh/<obj>.obj`
   - anchor bank: `MV-GoTrack/anchor_banks/<obj>.npz`
   - GoTrack checkpoint
4. output skip 확인.
   - `object_tracking/gotrack_output/world_pose_records.json`
   - `summary.json`
5. `capture_pc_status.json` 초기화.
   - 5개 PC 모두 key를 만든다.
   - daemon process 확인 결과를 `daemon_seen`에 기록한다.
   - 이 시점부터 monitor는 "not_started/daemon_missing"을 보여줄 수 있다.

### 7.2 Session start

1. `run_manifest.json` 작성.
2. `events.jsonl`에 `session_created` 기록.
3. gotrack daemon에 `init` command 전송.
4. `state.json.phase = "daemon_init"` 갱신.
5. `capture_pc_status.json`에 PC별 `init_sent=true`, `phase=init_sent` 기록.
6. robot tracker 생성.
7. gotrack daemon에 `start` command 전송.
8. `capture_pc_status.json`에 PC별 `start_sent=true`, `phase=start_sent` 기록.
9. `state.json.phase = "tracking"` 갱신.

### 7.3 Tracking loop

1. `GoTrackTracker.track(init_pose_world)`를 소비한다.
2. 성공 frame마다 `world_pose_records.jsonl`에 append한다.
3. `tracker.status`를 주기적으로 snapshot해 `state.json`을 atomic write한다.
4. `tracker.status.per_pc_last_frame`을 읽어 PC별 age/frame/frames_received를
   `capture_pc_status.json`에 atomic write한다.
5. PC별 `last_obs_age_s`가 threshold를 넘으면 `stale`로 표시한다.
6. 실패 reason aggregate는 `tracker.status.counts.fail_by_reason`에서 가져온다.
7. max frame, robot execution 종료 signal, 또는 사용자 stop 조건에 도달하면
   loop를 끝낸다.

주의: 현재 `GoTrackTracker.track()`은 성공 frame만 yield한다. 실패 frame을
개별 record로 남기려면 기존 코드를 수정하기보다 새 adapter/subclass에서
status 변화 감지를 통해 failure event를 기록하는 방식으로 시작한다.

### 7.4 Stop/finalize

1. gotrack daemon에 `stop` command 전송.
2. `capture_pc_status.json`에 PC별 `stop_sent=true`, `phase=stopping` 기록.
3. `world_pose_records.jsonl`을 읽어 `world_pose_records.json`을 atomic write한다.
4. `summary.json` 작성.
5. 성공 종료 시 5개 PC 상태를 `complete`로 바꾼다. 실패 종료 시 stale/command
   failure가 있던 PC만 `failed`로 남긴다.
6. `state.json.phase = "done"` 또는 `"failed"` 기록.
7. 필요하면 upload watcher를 실행하고 upload status를 갱신한다.

## 8. 완료된 작업과 남은 작업

### 완료된 작업

- capture PC별 GoTrack daemon 구현.
- robot PC GoTrack tracker 구현.
- prior pose publish와 anchor observation subscribe 구현.
- frame id 기반 sync buffer 구현.
- triangulation + Kabsch pose fitting 구현.
- live dashboard 구현.
- debug crop background upload 구현.
- 5개 capture PC daemon 관리 스크립트 구현.
- offline GoTrack + overlay batch에서 skip/resume/progress/upload 패턴 구현.
- FoundPose init orchestrator의 live progress 출력 구현.

### 남은 작업

- pull/merge 정리
  - 현재 repo divergence와 dirty worktree 때문에 자동 pull은 보류.
  - tracking 작업 전용 브랜치에서 원격 최신 반영 전략을 정해야 한다.

- live tracking session wrapper 추가
  - gotrack daemon `init/start/stop`.
  - `GoTrackTracker` lifecycle 관리.
  - trial output directory 관리.

- durable progress 추가
  - `events.jsonl`, `state.json`, `summary.json`.
  - process 재시작 후에도 누적 상태 확인 가능해야 한다.
  - `capture_pc_status.json`으로 5개 capture PC 각각의 진행 중/완료 상태를
    실시간으로 확인 가능해야 한다.

- live pose output 추가
  - `world_pose_records.jsonl` append.
  - final `world_pose_records.json` 생성.
  - 기존 overlay 코드가 읽는 path contract 유지.

- skip/resume 구현
  - 예전 버전으로 이미 생성된 `world_pose_records.json`은 재실행하지 않는다.
  - partial output은 별도 상태로 표시한다.

- monitor script 추가
  - `monitor_foundpose_onboard.py`와 유사한 파일 기반 monitor.

- upload 상태 기록
  - debug crop 외 trial-level tracking output upload 상태도 progress에 포함.

- validation
  - fake tracker unit test.
  - one-PC dry run.
  - 5-PC smoke test.
  - 실제 trial에서 overlay까지 확인.

## 9. 검증 계획

1. Progress writer unit test
   - event append, atomic state write, final summary write 검증.

2. Fake session dry run
   - capture PC 없이 synthetic pose를 몇 frame 기록해
     `world_pose_records.jsonl`, `world_pose_records.json`, `state.json` 형식을
     확인한다.
   - synthetic per-PC observation timestamp를 넣어 `capture_pc_status.json`의
     `running/stale/complete` 판정도 확인한다.

3. One-PC smoke test
   - capture PC 한 대만 사용해 daemon init/start/stop과 first observation을
     확인한다.

4. Five-PC smoke test
   - `capture1`, `capture2`, `capture3`, `capture5`, `capture6` 모두 daemon
     연결.
   - `per_pc_last_frame` age가 정상적으로 갱신되는지 확인한다.

5. Real trial integration test
   - `pose_world.npy`가 있는 trial에서 tracking session 실행.
   - `world_pose_records.json` 생성 확인.
   - `src/figure/overlay_lift.py` 또는 overlay pipeline으로 pose record 소비 확인.

6. Crash/restart test
   - tracking 중 강제 종료 후 `state.json`과 `events.jsonl`로 마지막 상태를
     확인한다.
   - `capture_pc_status.json`에서 어떤 PC가 마지막으로 frame을 보냈는지
     확인한다.
   - 완료된 trial은 `--force` 없이는 skip되는지 확인한다.

## 10. 권장 구현 순서

1. 새 progress writer 추가.
2. standalone `run_gotrack_session.py` 추가.
3. gotrack daemon command orchestration 연결.
4. pose record persistence 연결.
5. monitor script 추가.
6. one-PC, five-PC 순서로 smoke test.
7. 안정화 후 `run_auto.py`에 optional integration hook을 붙일지 결정.

이 순서면 기존 tracking core를 건드리지 않고도, 실험 중 "어디까지 진행됐는지"를
파일로 누적 확인할 수 있다.

## 11. 2026-07-01 capture PC 환경 재설정

### 11.1 실제 실패 원인

5개 PC 병렬 episode smoke test에서 scheduler 자체는 정상 동작했다. 각 worker는
episode를 claim했고, GoTrack도 카메라 24개를 찾고 model load까지 진행했다.
실패 지점은 DINOv2 feature extraction 내부의 xformers attention kernel 선택이다.

대표 진단 결과:

```text
torch 2.11.0+cu128
gpu NVIDIA GeForce RTX 5060 Ti
compute capability (12, 0)
xformers 0.0.35
build.torch_version 2.10.0+cu128
build.env.TORCH_CUDA_ARCH_LIST 7.5 8.0+PTX 8.0 9.0a
memory_efficient_attention.cutlassF-blackwell unavailable
```

따라서 문제는 object tracking 코드의 분산 스케줄링이 아니라, capture PC의
`gotrack_cu128` runtime이 Blackwell에서 xformers attention을 FP32 입력으로
호출하고 있다는 점이다. import는 성공하지만 실제 GoTrack이 사용하는 FP32
`memory_efficient_attention_forward` 호출에서 다음 형태로 실패한다.

```text
NotImplementedError: No operator found for memory_efficient_attention_forward
requires device with capability <= (9, 0) but your GPU has capability (12, 0)
```

이 문제는 fallback으로 xformers를 끄거나 CPU/native attention으로 우회해서
해결하지 않는다. Blackwell용 xformers attention은 FP16/BF16 경로를 사용해야
하므로, `gotrack_cu128` 안에서 현재 설치된 torch/CUDA ABI에 맞춰 xformers를 다시
빌드하고, GoTrack subprocess가 `--forward-precision bf16`으로 실행되도록 wrapper
환경을 맞춘다.

### 11.2 추가된 환경 스크립트

- `scripts/setup_gotrack_blackwell_xformers.sh`
  - 각 capture PC 내부에서 실행한다.
  - `gotrack_cu128` conda env를 activate한다.
  - build dependency와 CUDA 12.8 nvcc package를 확인/설치한다.
  - `TORCH_CUDA_ARCH_LIST=12.0`으로 xformers를 source build한다.
  - 마지막에 실제 실패 지점과 같은 shape의 BF16 xformers attention 호출을
    실행한다.

- `scripts/launch_gotrack_env_setup.sh`
  - robot PC에서 실행한다.
  - `capture1`, `capture2`, `capture3`, `capture5`, `capture6`에 SSH를 PC당 한 번만
    열어 setup을 background로 시작한다.
  - 이후 상태 확인은 SSH polling이 아니라 공유 로그 파일을 읽는다.

- `scripts/run_batch_object_overlay_with_env.py`
  - 원본 `src/process/batch_object_overlay.py`는 수정하지 않는다.
  - wrapper 내부에서 GoTrack tracking subprocess에만 BF16 xformers
    `sitecustomize` shim을 `PYTHONPATH`로 주입한다.
  - 기본값은 `AUTODEX_GOTRACK_FORWARD_PRECISION=bf16`이고, 필요 시 env var로
    override할 수 있다.
  - 현재 `run_multiview_gotrack_anchor_online.py`는 `--forward-precision` CLI를
    받지 않으므로 CLI 인자는 추가하지 않는다.
  - global autocast는 pose tensor까지 BF16으로 만들어 `numpy()` 변환을 깨뜨릴 수
    있으므로 사용하지 않는다. xformers attention 입력만 BF16으로 계산하고 출력은
    FP32로 되돌린다.

- `scripts/setup_object_overlay_env.sh`
  - overlay rendering은 GoTrack env가 아니라 보통 `~/anaconda3/envs/paradex`를
    사용한다.
  - `overlay_object_video_single.py`에는 `transforms3d`, `trimesh`,
    `nvdiffrast`가 필요하다.
  - 2026-07-01 smoke test에서 GoTrack은 완료됐지만 overlay가
    `ModuleNotFoundError: transforms3d` 또는 `ModuleNotFoundError: trimesh`로
    실패했으므로, capture PC마다 이 스크립트로 overlay env를 별도 검증한다.

완료 판정 기준:

```text
[verify] memory_efficient_attention_bf16_ok ...
[setup] done
```

위 두 줄이 각 PC의 setup log에 있어야 해당 PC 환경 재설정이 완료된 것이다.

### 11.3 실행 명령

한 PC에서 직접 검증:

```bash
bash scripts/setup_gotrack_blackwell_xformers.sh --verify-only
```

한 PC에서 실제 재빌드:

```bash
bash scripts/setup_gotrack_blackwell_xformers.sh
```

한 PC에서 overlay env 검증/설치:

```bash
bash scripts/setup_object_overlay_env.sh
```

5개 capture PC에 한 번씩만 SSH로 launch:

```bash
LOG_ID=gotrack_sm120_20260701 \
STAGGER_SECONDS=60 \
bash scripts/launch_gotrack_env_setup.sh launch
```

진행 상태 확인:

```bash
LOG_ID=gotrack_sm120_20260701 \
bash scripts/launch_gotrack_env_setup.sh status
```

이미 xformers source build가 끝난 PC에서 검증만 다시 실행:

```bash
PCS="capture3 capture5 capture6" \
LOG_ID=gotrack_sm120_20260701 \
SETUP_ARGS=--verify-only \
bash scripts/launch_gotrack_env_setup.sh launch
```

공유 로그 위치:

```text
/home/robot/shared_data/AutoDex/object_tracking/env_setup/<LOG_ID>/
  capture1.launch.log
  capture1.setup.log
  capture1.setup.pid
  capture2.launch.log
  capture2.setup.log
  ...
```

SSH reset이 발생한 PC는 `*.launch_failed` 파일로 남는다. 이 경우 전체를 다시
누르지 말고 실패한 PC만 대상으로 같은 `LOG_ID`를 유지해 launch한다.

```bash
PCS="capture2 capture5" \
LOG_ID=gotrack_sm120_20260701 \
bash scripts/launch_gotrack_env_setup.sh launch
```

### 11.4 환경 재설정 후 재실행

기존 5-PC smoke test schedule은 다음 위치에 있다.

```text
/home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_5pc_banana_20260701
```

환경 재설정이 끝나면 schedule을 새로 만들 필요 없이 실패 task만 retry한다.

```bash
python scripts/run_gotrack_episode_scheduler.py \
  --mode launch \
  --schedule-dir /home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_5pc_banana_20260701 \
  --pcs capture1 capture2 capture3 capture5 capture6 \
  --repo-dir /home/robot/AutoDex \
  --retry-failed \
  --ssh-tty-pcs capture1
```

`--retry-failed`는 launch 전에 실패로 끝난 task의 오래된 claim lock을 한 번만
정리한 뒤 worker들이 다시 claim하게 한다.
capture1처럼 non-TTY SSH command가 pre-auth 단계에서 reset되는 PC는
`--ssh-tty-pcs`에 넣어 `ssh -tt`로 launch한다.

대시보드는 기존과 같은 schedule directory를 보면 된다.

```bash
python scripts/gotrack_episode_dashboard.py \
  --schedule-dir /home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_5pc_banana_20260701 \
  --host 0.0.0.0 \
  --port 8768
```

2026-07-01 실제 5-PC banana smoke test 결과:

```text
schedule: /home/robot/shared_data/AutoDex/object_tracking/episode_scheduler/gotrack_5pc_banana_20260701
episodes: 5/5 done
object: allegro/banana
episodes:
  20260405_073417
  20260405_073554
  20260405_073727
  20260405_073843
  20260405_074229
outputs:
  /home/robot/shared_data/AutoDex/experiment/selected_100/allegro/banana/<episode>/object_tracking/gotrack_output/world_pose_records.json
  /home/robot/shared_data/AutoDex/object_overlay_video/allegro/banana/<episode>/overlay_<serial>.mp4
overlay files: 24 per episode
dashboard: http://127.0.0.1:8769/
```
