document.addEventListener("DOMContentLoaded", () => {
    const modelInput = document.getElementById("modelInput");
    const fileInput = document.getElementById("fileInput");
    const modelButtons = Array.from(document.querySelectorAll(".model-choice"));
    const requirementsCard = document.getElementById("requirementsCard");
    const requirementModality = document.getElementById("requirementModality");
    const requirementFormats = document.getElementById("requirementFormats");
    const requirementGuidance = document.getElementById("requirementGuidance");
    const outputGuidance = document.getElementById("outputGuidance");
    const previewShell = document.getElementById("previewShell");
    const previewImage = document.getElementById("previewImage");
    const previewHint = document.getElementById("previewHint");
    const filePrompt = document.getElementById("filePrompt");
    const fileMeta = document.getElementById("fileMeta");
    const dropzone = document.getElementById("dropzone");
    const submitButton = document.getElementById("submitButton");
    const sampleButton = document.getElementById("sampleButton");
    const sampleModelInput = document.getElementById("sampleModelInput");
    const sampleForm = document.getElementById("sampleForm");
    const analysisForm = document.getElementById("analysisForm");
    const initialCardData = document.getElementById("initialCardData");

    if (!modelInput || !fileInput || !analysisForm) {
        return;
    }

    const cards = {};
    modelButtons.forEach((button) => {
        cards[button.dataset.model] = {
            key: button.dataset.model,
            accept: button.dataset.accept,
            formats: button.dataset.formats,
            modality: button.dataset.modality,
            guidance: button.dataset.guidance,
            output: button.dataset.output,
            sample: button.dataset.sample === "yes",
            badge: button.querySelector(".model-badge")?.textContent?.trim() || "Study",
        };
    });

    const resetPreview = (hintText) => {
        previewShell?.classList.add("is-empty");
        previewImage?.removeAttribute("src");
        if (previewHint) {
            previewHint.textContent = hintText;
        }
    };

    const applyCard = (card) => {
        if (!card) {
            return;
        }

        modelInput.value = card.key;
        if (sampleModelInput) {
            sampleModelInput.value = card.key;
        }
        fileInput.accept = card.accept;
        requirementModality.textContent = card.modality;
        requirementFormats.textContent = card.formats;
        requirementGuidance.textContent = card.guidance;
        outputGuidance.textContent = card.output;

        modelButtons.forEach((button) => {
            button.classList.toggle("is-selected", button.dataset.model === card.key);
        });

        if (sampleButton) {
            if (card.sample) {
                sampleButton.disabled = false;
                sampleButton.classList.remove("is-disabled");
                sampleButton.textContent = `Use Sample ${card.badge}`;
                sampleForm?.setAttribute("action", `/run-sample/${card.key}`);
            } else {
                sampleButton.disabled = true;
                sampleButton.classList.add("is-disabled");
                sampleButton.textContent = "No Built-in Sample";
                sampleForm?.setAttribute("action", "#");
            }
        }

        filePrompt.textContent = card.key === "atrium"
            ? "Upload a .nii or .nii.gz cardiac MRI volume"
            : "Upload a chest X-ray image";
        fileMeta.textContent = `Accepted format: ${card.formats}`;

        const hintText = card.key === "atrium"
            ? "MRI NIfTI uploads are validated server-side. After analysis, the extracted middle slice and segmentation overlay will appear here."
            : "Chest X-ray images preview here before the model runs.";
        resetPreview(hintText);
        fileInput.value = "";
    };

    const renderPreview = (file) => {
        if (!file) {
            const activeCard = cards[modelInput.value];
            const hintText = activeCard?.key === "atrium"
                ? "MRI NIfTI uploads are validated server-side. After analysis, the extracted middle slice and segmentation overlay will appear here."
                : "Chest X-ray images preview here before the model runs.";
            resetPreview(hintText);
            filePrompt.textContent = activeCard?.key === "atrium"
                ? "Upload a .nii or .nii.gz cardiac MRI volume"
                : "Upload a chest X-ray image";
            fileMeta.textContent = activeCard ? `Accepted format: ${activeCard.formats}` : "";
            return;
        }

        filePrompt.textContent = file.name;
        fileMeta.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB selected`;

        if (modelInput.value === "atrium" || !file.type.startsWith("image/")) {
            resetPreview("The file is ready for upload. MRI volumes are previewed after server-side slice extraction.");
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewShell.classList.remove("is-empty");
        };
        reader.readAsDataURL(file);
    };

    modelButtons.forEach((button) => {
        button.addEventListener("click", () => {
            applyCard(cards[button.dataset.model]);
        });
    });

    fileInput.addEventListener("change", () => {
        renderPreview(fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone?.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("is-dragging");
        });
    });

    ["dragleave", "dragend", "drop"].forEach((eventName) => {
        dropzone?.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("is-dragging");
        });
    });

    dropzone?.addEventListener("drop", (event) => {
        const files = event.dataTransfer.files;
        if (!files || !files.length) {
            return;
        }

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(files[0]);
        fileInput.files = dataTransfer.files;
        renderPreview(files[0]);
    });

    analysisForm.addEventListener("submit", () => {
        submitButton.disabled = true;
        submitButton.textContent = "Running Analysis...";
    });

    if (initialCardData) {
        applyCard({
            key: initialCardData.dataset.model,
            accept: initialCardData.dataset.accept,
            formats: initialCardData.dataset.formats,
            modality: initialCardData.dataset.modality,
            guidance: initialCardData.dataset.guidance,
            output: initialCardData.dataset.output,
            sample: initialCardData.dataset.sample === "yes",
            badge: cards[initialCardData.dataset.model]?.badge || "Study",
        });
    }
});
