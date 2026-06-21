/* ===================== AutoDex page interactions ===================== */
document.addEventListener('DOMContentLoaded', function () {

  /* ---- Transparent nav over the full-screen hero, solid after scroll ---- */
  const navEl = document.querySelector('.nav');
  const fvEl = document.querySelector('.fullvideo');
  if (navEl) {
    const onNavScroll = () => {
      const threshold = (fvEl ? fvEl.offsetHeight : 300) - 70;
      navEl.classList.toggle('solid', window.scrollY > threshold);
    };
    window.addEventListener('scroll', onNavScroll, { passive: true });
    window.addEventListener('resize', onNavScroll);
    onNavScroll();
  }

  /* ---- Interactive showcase: pick object -> generate -> execute ---- */
  const OBJECTS = [
    { name: 'Donut',        gen: 'static/videos/gen_donut.mp4',             exec: 'static/videos/result_donut.mp4',             poster: 'static/posters/result_donut.jpg' },
    { name: 'Blue vase',    gen: 'static/videos/gen_blue_vase.mp4',         exec: 'static/videos/result_bluevase.mp4',          poster: 'static/posters/result_bluevase.jpg' },
    { name: 'Brush',        gen: 'static/videos/gen_beige_brush.mp4',       exec: 'static/videos/result_beige_brush.mp4',       poster: 'static/posters/result_beige_brush.jpg' },
    { name: 'Small bowl',   gen: 'static/videos/gen_smallbowl.mp4',         exec: 'static/videos/result_smallbowl.mp4',         poster: 'static/posters/result_smallbowl.jpg' },
    { name: 'Serving bowl', gen: 'static/videos/gen_servingbowl_small.mp4', exec: 'static/videos/result_servingbowl_small.mp4', poster: 'static/posters/result_servingbowl_small.jpg' },
  ];
  const scVideo = document.getElementById('sc-video');
  const scObjs = document.getElementById('sc-objs');
  const scSteps = document.querySelectorAll('.sc-step');
  if (scVideo && scObjs) {
    let curObj = 0, curStep = 'gen';

    // build object selector
    OBJECTS.forEach((o, i) => {
      const b = document.createElement('button');
      b.className = 'sc-obj' + (i === 0 ? ' active' : '');
      b.innerHTML = '<img src="' + o.poster + '" alt="' + o.name + '"><span>' + o.name + '</span>';
      b.addEventListener('click', () => selectObject(i));
      scObjs.appendChild(b);
    });

    function setStepUI(step) {
      scSteps.forEach(s => s.classList.toggle('active', s.dataset.step === step));
    }
    function playClip(step) {
      curStep = step;
      const o = OBJECTS[curObj];
      scVideo.src = (step === 'gen' ? o.gen : o.exec);
      scVideo.poster = o.poster;
      setStepUI(step);
      scVideo.load();
      scVideo.play().catch(() => {});
    }
    function selectObject(i) {
      curObj = i;
      document.querySelectorAll('.sc-obj').forEach((b, j) => b.classList.toggle('active', j === i));
      playClip('gen');
    }
    // auto-advance: gen -> exec -> (loop) gen
    scVideo.addEventListener('ended', () => {
      playClip(curStep === 'gen' ? 'exec' : 'gen');
    });
    scSteps.forEach(s => s.addEventListener('click', () => playClip(s.dataset.step)));

    // init
    selectObject(0);
    // pause when off-screen, resume when back
    new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) scVideo.play().catch(()=>{}); else scVideo.pause(); });
    }, { threshold: 0.2 }).observe(scVideo);
  }

  /* ---- Pipeline stepper ---- */
  const tabs = document.querySelectorAll('.step-tab');
  const panels = document.querySelectorAll('.step-panel');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const id = tab.dataset.step;
      tabs.forEach(t => t.classList.toggle('active', t === tab));
      panels.forEach(p => {
        const on = p.dataset.step === id;
        p.classList.toggle('active', on);
        const v = p.querySelector('video');
        if (v) { if (on) v.play().catch(()=>{}); else v.pause(); }
      });
    });
  });

  /* ---- Object gallery ---- */
  const mainVid = document.getElementById('gallery-video');
  const mainName = document.getElementById('gallery-name');
  document.querySelectorAll('.thumb').forEach(th => {
    th.addEventListener('click', () => {
      document.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
      th.classList.add('active');
      if (mainVid) {
        mainVid.querySelector('source').src = th.dataset.src;
        mainVid.poster = th.dataset.poster;
        mainVid.load(); mainVid.play().catch(()=>{});
      }
      if (mainName) mainName.textContent = th.dataset.name;
    });
  });

  /* ---- Active nav link on scroll ---- */
  const navLinks = document.querySelectorAll('.nav-links a[data-sec]');
  const secs = Array.from(navLinks).map(a => document.getElementById(a.dataset.sec)).filter(Boolean);
  if (secs.length) {
    const navObs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) navLinks.forEach(a => a.classList.toggle('active', a.dataset.sec === e.target.id));
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    secs.forEach(s => navObs.observe(s));
  }

  /* ---- Copy BibTeX ---- */
  const copyBtn = document.getElementById('copy-bib');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(document.getElementById('bibtex-text').textContent).then(() => {
        copyBtn.textContent = 'Copied'; setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
      });
    });
  }

  /* ---- Pause off-screen videos (perf) ---- */
  const vidObs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      const v = e.target;
      if (e.isIntersecting) { if (v.dataset.auto !== 'off') v.play().catch(()=>{}); }
      else v.pause();
    });
  }, { threshold: 0.25 });
  document.querySelectorAll('video.lazyplay').forEach(v => vidObs.observe(v));
});
