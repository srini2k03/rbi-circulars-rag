/* sb-kit.js: theme toggle, cursor spotlight and scroll reveal for the project sites.
   1. Put this inline in <head>, BEFORE any stylesheet paints, to avoid a theme flash:
        <script>(function(){try{var t=localStorage.getItem('theme');document.documentElement.dataset.theme=(t==='light'||t==='dark')?t:'dark';}catch(e){document.documentElement.dataset.theme='dark';}document.documentElement.classList.remove('no-js');})();</script>
      and give <html> the class "no-js" in the markup.
   2. Load this file with <script src="sb-kit.js" defer></script>.
   3. Theme toggle: any element with [data-theme-toggle] flips light/dark (same localStorage key "theme"
      as the portfolio, so the choice carries across Srinivasan's sites on github.io).
   Everything here is progressive enhancement: the page must work without it. */
(function () {
  "use strict";
  var root = document.documentElement;
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function setTheme(t) {
    root.dataset.theme = t;
    try { localStorage.setItem("theme", t); } catch (e) { /* storage blocked: theme still applies for this page */ }
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
      b.setAttribute("aria-pressed", t === "dark" ? "true" : "false");
    });
    document.dispatchEvent(new CustomEvent("sb:theme", { detail: t }));
  }

  function init() {
    setTheme(root.dataset.theme === "light" ? "light" : "dark");
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.addEventListener("click", function () { setTheme(root.dataset.theme === "dark" ? "light" : "dark"); });
    });

    // Cursor spotlight on cards.
    if (!reduced && window.matchMedia("(pointer: fine)").matches) {
      document.addEventListener("pointermove", function (e) {
        var el = e.target.closest && e.target.closest(".sb-spotlight");
        if (!el) return;
        var r = el.getBoundingClientRect();
        el.style.setProperty("--mx", (e.clientX - r.left) + "px");
        el.style.setProperty("--my", (e.clientY - r.top) + "px");
      }, { passive: true });
    }

    // Reveal on scroll.
    var items = document.querySelectorAll(".sb-reveal");
    if (reduced || !("IntersectionObserver" in window)) {
      items.forEach(function (el) { el.classList.add("is-in"); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); }
        });
      }, { rootMargin: "0px 0px -8% 0px" });
      items.forEach(function (el) { io.observe(el); });
      // Never leave content hidden: reveal anything still hidden after 2.5 s (e.g. print, tall viewports).
      setTimeout(function () { items.forEach(function (el) { el.classList.add("is-in"); }); }, 2500);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
