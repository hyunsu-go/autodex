/* ===================== AutoDex page interactions ===================== */
document.addEventListener('DOMContentLoaded', function () {

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
