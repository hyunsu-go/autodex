/* ===================== AutoDex page interactions ===================== */
document.addEventListener('DOMContentLoaded', function () {

  /* ---- Scroll reveal ---- */
  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revealObs.unobserve(e.target); } });
  }, { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach(el => revealObs.observe(el));

  /* ---- Count-up stats ---- */
  function animateCount(el) {
    const target = parseFloat(el.dataset.target);
    const dec = (el.dataset.dec | 0);
    const suffix = el.dataset.suffix || '';
    const dur = 1400; const start = performance.now();
    function tick(now) {
      const p = Math.min((now - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = target * eased;
      el.textContent = val.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }
  const countObs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { animateCount(e.target); countObs.unobserve(e.target); } });
  }, { threshold: 0.5 });
  document.querySelectorAll('[data-target]').forEach(el => countObs.observe(el));

  /* ---- Comparison bars ---- */
  const barObs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.style.width = e.target.dataset.w + '%'; barObs.unobserve(e.target); } });
  }, { threshold: 0.5 });
  document.querySelectorAll('.cmp-fill').forEach(el => barObs.observe(el));

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
        if (v) { if (on) { v.play().catch(()=>{}); } else { v.pause(); } }
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
  const navObs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        navLinks.forEach(a => a.classList.toggle('active', a.dataset.sec === e.target.id));
      }
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach(s => navObs.observe(s));

  /* ---- Copy BibTeX ---- */
  const copyBtn = document.getElementById('copy-bib');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const txt = document.getElementById('bibtex-text').textContent;
      navigator.clipboard.writeText(txt).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1600);
      });
    });
  }

  /* ---- Lazy-play videos only when visible (perf) ---- */
  const vidObs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      const v = e.target;
      if (e.isIntersecting) { if (v.dataset.auto !== 'off') v.play().catch(()=>{}); }
      else v.pause();
    });
  }, { threshold: 0.25 });
  document.querySelectorAll('video.lazyplay').forEach(v => vidObs.observe(v));
});
