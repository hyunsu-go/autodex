/* ===================== AutoDex page interactions ===================== */
document.addEventListener('DOMContentLoaded', function () {

  /* ---- Teaser video chapter timeline ----
     PLACEHOLDER timings (seconds) — replace `t` with the real chapter starts. */
  const CHAPTERS = [
    { t: 0,  label: 'Generate' },
    { t: 5,  label: 'Select' },
    { t: 9,  label: 'Execute' },
    { t: 15, label: 'Reset' },
  ];
  const video = document.getElementById('teaser-video');
  const segWrap = document.getElementById('vt-segs');
  const labWrap = document.getElementById('vt-labels');
  if (video && segWrap && labWrap) {
    const seek = (t) => { try { video.currentTime = t + 0.01; } catch(e){} video.play().catch(()=>{}); };
    const chapEnd = (i, dur) => (i < CHAPTERS.length - 1 ? CHAPTERS[i + 1].t : dur);

    function build() {
      const dur = video.duration || (CHAPTERS[CHAPTERS.length - 1].t + 5);
      segWrap.innerHTML = ''; labWrap.innerHTML = '';
      CHAPTERS.forEach((c, i) => {
        const span = Math.max(0.5, chapEnd(i, dur) - c.t);
        const seg = document.createElement('div');
        seg.className = 'vt-seg'; seg.style.flexGrow = span; seg.style.flexBasis = '0';
        seg.innerHTML = '<span class="fill"></span>';
        seg.addEventListener('click', () => seek(c.t));
        segWrap.appendChild(seg);

        const lab = document.createElement('button');
        lab.className = 'vt-lab'; lab.style.flexGrow = span;
        lab.innerHTML = '<span class="n">' + (i + 1) + '</span><span>' + c.label + '</span>';
        lab.addEventListener('click', () => seek(c.t));
        labWrap.appendChild(lab);
      });
    }
    function update() {
      const dur = video.duration; if (!dur) return;
      const t = video.currentTime;
      let active = 0;
      for (let i = 0; i < CHAPTERS.length; i++) if (t >= CHAPTERS[i].t) active = i;
      const segs = segWrap.children, labs = labWrap.children;
      CHAPTERS.forEach((c, i) => {
        const span = chapEnd(i, dur) - c.t;
        const pct = Math.max(0, Math.min(1, (t - c.t) / (span || 1)));
        if (segs[i]) { segs[i].querySelector('.fill').style.width = (pct * 100) + '%'; segs[i].classList.toggle('active', i === active); }
        if (labs[i]) labs[i].classList.toggle('active', i === active);
      });
    }
    video.addEventListener('loadedmetadata', build);
    video.addEventListener('timeupdate', update);
    if (video.readyState >= 1) { build(); update(); }
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
