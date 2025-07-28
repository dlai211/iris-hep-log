const toggleBtn = document.getElementById("sidebar-toggle");
const sidebar = document.getElementById("sidebar");

toggleBtn.onclick = function () {
    sidebar.classList.toggle("open");

    const icon = toggleBtn.querySelector("i");
    if (sidebar.classList.contains("open")) {
        icon.classList.remove("fa-angle-right");
        icon.classList.add("fa-angle-left");
        toggleBtn.style.left = "25px"; // offset from sidebar
    } else {
        icon.classList.remove("fa-angle-left");
        icon.classList.add("fa-angle-right");
        toggleBtn.style.left = "10px";
    }
};

// Optional: auto-close sidebar on outside click
document.addEventListener("click", function (event) {
    const isClickInside = sidebar.contains(event.target) || toggleBtn.contains(event.target);
    if (!isClickInside && sidebar.classList.contains("open")) {
        sidebar.classList.remove("open");
        const icon = toggleBtn.querySelector("i");
        icon.classList.remove("fa-angle-left");
        icon.classList.add("fa-angle-right");
        toggleBtn.style.left = "10px";
    }
});

document.querySelectorAll("#sidebar a").forEach(link => {
    const currentUrl = window.location.href;
    const linkHref = new URL(link.href, document.baseURI).href;

    if (currentUrl === linkHref) {
      link.classList.add("active");
    }
    link.addEventListener("click", () => {
        // window.location.reload();
        const href = link.getAttribute("href");
        window.location.href = href;
        window.location.reload();
    });
});
// document.querySelectorAll("#sidebar a").forEach(link => {
//     link.addEventListener("click", (e) => {
//         e.preventDefault();  // stop default link behavior
//         const href = link.getAttribute("href");

//         // Set location to force reload even if hash is same
//         window.location.href = href;
//         window.location.reload();  // ✅ force full reload
//     });
// });
