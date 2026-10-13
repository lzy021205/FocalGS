const dialog = document.getElementById("imageDialog");
const dialogImage = document.getElementById("dialogImage");
const dialogCaption = document.getElementById("dialogCaption");
const closeButton = dialog ? dialog.querySelector(".dialog-close") : null;

document.querySelectorAll(".image-open").forEach((button) => {
  button.addEventListener("click", () => {
    const src = button.dataset.full;
    const caption = button.dataset.caption || "";
    const previewImage = button.querySelector("img");

    if (!dialog || !dialog.showModal || !src) {
      return;
    }

    dialogImage.src = src;
    dialogImage.alt = previewImage ? previewImage.alt : caption;
    dialogCaption.textContent = caption;
    dialog.showModal();
  });
});

if (closeButton) {
  closeButton.addEventListener("click", () => dialog.close());
}

if (dialog) {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      dialog.close();
    }
  });

  dialog.addEventListener("close", () => {
    dialogImage.removeAttribute("src");
  });
}

document.querySelectorAll("video").forEach((video) => {
  video.muted = true;
  video.playsInline = true;
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {
      video.setAttribute("controls", "controls");
    });
  }
});

document.querySelectorAll("[data-focal-comparison]").forEach((component) => {
  const tabs = Array.from(component.querySelectorAll("[data-focal-target]"));
  const panels = Array.from(component.querySelectorAll("[data-focal-panel]"));

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.focalTarget;

      tabs.forEach((item) => {
        const selected = item === tab;
        item.classList.toggle("active", selected);
        item.setAttribute("aria-selected", selected ? "true" : "false");
      });

      panels.forEach((panel) => {
        panel.hidden = panel.dataset.focalPanel !== target;
        panel.classList.toggle("active", panel.dataset.focalPanel === target);
      });
    });
  });
});
