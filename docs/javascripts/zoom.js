/* Make every figure open at full resolution in a new tab.
 *
 * The screenshots are larger than any column the site can give them, so the
 * page shows a scaled copy. Clicking one opens the original.
 *
 * document$ is Material's per-page observable: it fires again after each
 * instant-navigation page change, where a plain DOMContentLoaded would not.
 */
document$.subscribe(function () {
  document.querySelectorAll(".md-typeset img").forEach(function (img) {
    if (img.closest("a")) {
      return;                       // already linked, leave it alone
    }
    var link = document.createElement("a");
    link.className = "amz-zoom";
    link.href = img.getAttribute("src");
    link.target = "_blank";
    link.rel = "noopener";
    link.title = "Open the full-size image";
    img.parentNode.insertBefore(link, img);
    link.appendChild(img);
  });
});
