// SPDX-FileCopyrightText: 2026 Qoro Quantum
// SPDX-License-Identifier: GPL-3.0-or-later

// Makes the "On this page" highlight follow the link you click.
//
// The theme's scroll-spy marks the last entry as current whenever the page is
// scrolled to the bottom. On a page whose last sections are shorter than the
// window, clicking one of them scrolls to the bottom, so the last entry was
// marked instead. This adds just enough room after the footer for the last
// entry's heading to reach the top of the window, and none on pages that
// already have it.
(() => {
  const links = document.querySelectorAll(".toc-drawer .toc-tree a[href^='#']");
  const last = links[links.length - 1];
  const target = last?.hash && document.getElementById(decodeURIComponent(last.hash.slice(1)));
  const content = document.querySelector(".content");
  if (!target || !content) return;

  const room = document.createElement("div");
  room.setAttribute("aria-hidden", "true");
  content.append(room);
  const top = (element) => element.getBoundingClientRect().top + window.scrollY;
  const fit = () => {
    // The room's top is where the page's content ends.
    const below = top(room) - top(target);
    room.style.height = `${Math.max(0, window.innerHeight - below)}px`;
  };
  fit();
  // Refit when images load or the window changes size.
  new ResizeObserver(fit).observe(document.querySelector("article") ?? content);
  window.addEventListener("resize", fit);
})();
