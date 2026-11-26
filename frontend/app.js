// Configuration & Endpoints
const API_BASE = "";

// DOM Elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const hwDevice = document.getElementById("hwDevice");
const modelStatus = document.getElementById("modelStatus");

// Navigation & Mobile Drawer
const navItems = document.querySelectorAll(".nav-item");
const tabPages = document.querySelectorAll(".tab-page");
const sidebar = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const mobileDrawerBtn = document.getElementById("openDrawerBtn");
const closeSidebarBtn = document.getElementById("closeSidebarBtn");
const bottomNavHardwareBtn = document.getElementById("bottomNavHardwareBtn");
const mobileStatusPill = document.getElementById("mobileStatusPill");
const mobileStatusDot = document.getElementById("mobileStatusDot");
const mobileStatusText = document.getElementById("mobileStatusText");

// Recorder Elements
const recordBtn = document.getElementById("recordBtn");
const recordIcon = document.getElementById("recordIcon");
const recordingTimer = document.getElementById("recordingTimer");
const recorderHint = document.getElementById("recorderHint");
const audioPreviewContainer = document.getElementById("audioPreviewContainer");
const sampleAudioPreview = document.getElementById("sampleAudioPreview");
const sampleNameInput = document.getElementById("sampleNameInput");
const saveSampleBtn = document.getElementById("saveSampleBtn");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const mobileMicInput = document.getElementById("mobileMicInput");

// Generator Elements
const selectSample = document.getElementById("selectSample");
const textPrompt = document.getElementById("textPrompt");
const selectLang = document.getElementById("selectLang");
const generateClipBtn = document.getElementById("generateClipBtn");
const clipLoader = document.getElementById("clipLoader");
const resultCard = document.getElementById("resultCard");
const resultAudioPlayer = document.getElementById("resultAudioPlayer");
const downloadResultBtn = document.getElementById("downloadResultBtn");

// Library Grids
const samplesGrid = document.getElementById("samplesGrid");
const clipsGrid = document.getElementById("clipsGrid");

// Toast
const toast = document.getElementById("toast");
const toastIcon = document.getElementById("toastIcon");
const toastMsg = document.getElementById("toastMsg");

// Audio Recorder State
let mediaRecorder = null;
let audioChunks = [];
let recordInterval = null;
let recordSeconds = 0;
let currentRecordedBlob = null;

const downloadModelBtn = document.getElementById("downloadModelBtn");

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initRecorder();
    initDropzone();
    initGenerator();
    loadSettings();
    initModelInstaller();
    checkBackendStatus();
    loadSamples();
    loadClips();

    // Health check poll every 10s
    setInterval(checkBackendStatus, 10000);
});

function initModelInstaller() {
    if (!downloadModelBtn) return;
    downloadModelBtn.addEventListener("click", async () => {
        const modelLoaderBox = document.getElementById("modelLoaderBox");
        const fill = document.getElementById("modelProgressBarFill");
        const percent = document.getElementById("modelProgressPercent");
        const statusText = document.getElementById("modelLoaderStatus");

        let modelTimer = null;
        let progress = 3;

        try {
            downloadModelBtn.disabled = true;
            downloadModelBtn.classList.add("hidden");
            if (modelLoaderBox) modelLoaderBox.classList.remove("hidden");

            if (fill) fill.style.width = "3%";
            if (percent) percent.innerText = "0%";
            if (statusText) statusText.innerText = "Verificando modelos en ./backend/models/...";

            // Simular progreso suave del 0% al 94% durante la carga
            const intervalMs = 200;
            const expectedMs = 9000;
            const step = (90 / (expectedMs / intervalMs));

            modelTimer = setInterval(() => {
                if (progress < 94) {
                    const damping = progress > 75 ? 0.35 : (progress > 50 ? 0.7 : 1.0);
                    progress += step * damping;
                    if (progress > 94) progress = 94;

                    const pInt = Math.round(progress);
                    if (fill) fill.style.width = `${pInt}%`;
                    if (percent) percent.innerText = `${pInt}%`;

                    if (pInt < 25) {
                        if (statusText) statusText.innerText = "Verificando modelos locales...";
                    } else if (pInt < 55) {
                        if (statusText) statusText.innerText = "Cargando pesos de XTTS-v2 en RAM...";
                    } else if (pInt < 80) {
                        if (statusText) statusText.innerText = "Inicializando vocoder y tokenizer...";
                    } else {
                        if (statusText) statusText.innerText = "Optimizando tensores para inferencia...";
                    }
                }
            }, intervalMs);

            showToast("Iniciando carga del modelo local en ./backend/models/...", "info");

            const res = await fetch(`${API_BASE}/api/models/download`, { method: "POST" });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Error al inicializar el modelo");
            }

            const data = await res.json();

            // Completar al 100%
            if (modelTimer) clearInterval(modelTimer);
            if (fill) fill.style.width = "100%";
            if (percent) percent.innerText = "100%";
            if (statusText) statusText.innerText = "¡Modelo cargado y listo en memoria!";

            // Pausa para apreciar el 100%
            await new Promise(r => setTimeout(r, 650));

            showToast(data.message, "success");
            checkBackendStatus();

        } catch (err) {
            if (modelTimer) clearInterval(modelTimer);
            showToast(err.message, "error");
        } finally {
            if (modelTimer) clearInterval(modelTimer);
            if (modelLoaderBox) modelLoaderBox.classList.add("hidden");
            downloadModelBtn.classList.remove("hidden");
            downloadModelBtn.disabled = false;
        }
    });
}


// Toast notification helper
function showToast(message, type = "info") {
    toastMsg.innerText = message;
    toastIcon.className = type === "error" ? "fa-solid fa-circle-exclamation text-danger" : 
                         (type === "success" ? "fa-solid fa-circle-check text-success" : "fa-solid fa-info-circle");
    toast.classList.remove("hidden");
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 4000);
}

// Check Backend API Status
async function checkBackendStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        if (!res.ok) throw new Error("Offline");
        const data = await res.json();

        statusDot.className = "status-dot online";
        statusText.innerText = "API Conectada";
        hwDevice.innerHTML = `<i class="fa-solid fa-microchip"></i> ${data.device}`;
        
        if (data.model_loaded) {
            modelStatus.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> Cargado en Memoria`;
            if (downloadModelBtn) {
                downloadModelBtn.innerHTML = `<i class="fa-solid fa-circle-check"></i> Modelo Listo en Memoria`;
                downloadModelBtn.classList.remove("btn-primary");
                downloadModelBtn.classList.add("btn-outline");
            }
        } else if (data.model_downloaded) {
            modelStatus.innerHTML = `<i class="fa-solid fa-hard-drive"></i> Descargado (Local)`;
            if (downloadModelBtn) {
                downloadModelBtn.innerHTML = `<i class="fa-solid fa-bolt"></i> Cargar Modelo en Memoria`;
                downloadModelBtn.classList.remove("btn-outline");
                downloadModelBtn.classList.add("btn-primary");
            }
        } else {
            modelStatus.innerHTML = `<i class="fa-solid fa-download"></i> Pendiente Descarga`;
            if (downloadModelBtn) {
                downloadModelBtn.innerHTML = `<i class="fa-solid fa-cloud-arrow-down"></i> Inicializar Modelo Local`;
                downloadModelBtn.classList.remove("btn-outline");
                downloadModelBtn.classList.add("btn-primary");
            }
        }

        if (data.rvc_models) {
            updateRvcModelsDropdown(data.rvc_models);
        }

        // Sincronizar indicador de barra móvil
        if (mobileStatusDot) mobileStatusDot.className = "status-dot online";
        if (mobileStatusText) {
            mobileStatusText.innerText = data.model_loaded ? "IA Lista" : (data.model_downloaded ? "Descargado" : "Conectado");
        }
        if (mobileStatusPill) {
            mobileStatusPill.title = `Hardware: ${data.device} | Modelo: ${data.model_loaded ? 'Listo en Memoria' : (data.model_downloaded ? 'Descargado' : 'Pendiente')}`;
        }
    } catch (err) {
        statusDot.className = "status-dot";
        statusText.innerText = "Desconectado";
        hwDevice.innerText = "N/A";
        modelStatus.innerText = "N/A";

        if (mobileStatusDot) mobileStatusDot.className = "status-dot";
        if (mobileStatusText) mobileStatusText.innerText = "Desconectado";
    }
}

// Mobile Drawer Handlers
function openSidebarDrawer() {
    if (!sidebar) return;
    sidebar.classList.add("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.add("active");
    document.body.style.overflow = "hidden";
}

function closeSidebarDrawer() {
    if (!sidebar) return;
    sidebar.classList.remove("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
    document.body.style.overflow = "";
}

// Switch Active Tab (synchronized between desktop sidebar and mobile bottom nav)
function switchTab(targetTab) {
    if (!targetTab) return;

    // Synchronize desktop nav buttons
    document.querySelectorAll(".nav-item").forEach(n => {
        n.classList.toggle("active", n.getAttribute("data-tab") === targetTab);
    });

    // Synchronize mobile bottom nav buttons
    document.querySelectorAll(".bottom-nav-item[data-tab]").forEach(b => {
        b.classList.toggle("active", b.getAttribute("data-tab") === targetTab);
    });

    // Switch visible tab panel
    tabPages.forEach(p => p.classList.remove("active"));
    const activePanel = document.getElementById(`tab-${targetTab}`);
    if (activePanel) {
        activePanel.classList.add("active");
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    // Auto-close mobile drawer if opened
    closeSidebarDrawer();

    if (targetTab === "samples") loadSamples();
    if (targetTab === "clips") loadClips();
}

// Navigation Tabs & Mobile Drawer Initialization
function initNavigation() {
    // Sidebar nav items
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    // Mobile bottom nav items
    const bottomNavButtons = document.querySelectorAll(".bottom-nav-item[data-tab]");
    bottomNavButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    // Drawer Open Triggers
    if (mobileDrawerBtn) mobileDrawerBtn.addEventListener("click", openSidebarDrawer);
    if (bottomNavHardwareBtn) bottomNavHardwareBtn.addEventListener("click", openSidebarDrawer);
    if (mobileStatusPill) mobileStatusPill.addEventListener("click", openSidebarDrawer);

    // Drawer Close Triggers
    if (closeSidebarBtn) closeSidebarBtn.addEventListener("click", closeSidebarDrawer);
    if (sidebarBackdrop) sidebarBackdrop.addEventListener("click", closeSidebarDrawer);

    // Escape key closes mobile drawer
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && sidebar && sidebar.classList.contains("open")) {
            closeSidebarDrawer();
        }
    });
}

// Helper: Verificar si el navegador soporta getUserMedia en el contexto actual
function hasLiveAudioSupport() {
    return !!(navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function");
}

// Microphone Recorder
function initRecorder() {
    const canLiveRecord = hasLiveAudioSupport();

    if (!canLiveRecord) {
        // En conexiones HTTP sobre LAN (Wi-Fi), los navegadores móviles restringen getUserMedia por seguridad.
        // Habilitamos automáticamente la Grabadora Nativa del Celular.
        recorderHint.innerHTML = `
            <span style="color: #fbbf24; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
                <i class="fa-solid fa-mobile-screen"></i> Modo Grabadora Móvil:
            </span>
            Toca el micrófono para abrir la grabadora de tu teléfono.
        `;
    }

    recordBtn.addEventListener("click", async () => {
        // Si no hay soporte WebRTC (móvil en HTTP o sin permisos), activar grabadora nativa del teléfono
        if (!hasLiveAudioSupport()) {
            if (mobileMicInput) {
                mobileMicInput.click();
            } else if (fileInput) {
                fileInput.click();
            }
            return;
        }

        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            startRecording();
        } else {
            stopRecording();
        }
    });

    // Captura de audio desde la grabadora nativa del móvil
    if (mobileMicInput) {
        mobileMicInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                const file = e.target.files[0];
                currentRecordedBlob = file;
                const audioUrl = URL.createObjectURL(file);
                sampleAudioPreview.src = audioUrl;
                audioPreviewContainer.classList.remove("hidden");

                // Generar nombre sugerido por timestamp
                const now = new Date();
                const pad = (n) => String(n).padStart(2, '0');
                const timeStr = `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
                sampleNameInput.value = `VozMovil_${timeStr}`;

                recorderHint.innerHTML = `
                    <span style="color: #34d399; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
                        <i class="fa-solid fa-circle-check"></i> ¡Audio grabado con éxito!
                    </span>
                    Escúchalo abajo y presiona "Guardar Muestra".
                `;
                showToast("Audio grabado en el móvil cargado correctamente.", "success");
            }
        });
    }

    saveSampleBtn.addEventListener("click", uploadRecordedSample);
}

async function startRecording() {
    audioChunks = [];
    recordSeconds = 0;
    recordingTimer.innerText = "00:00";

    // Si el contexto no admite getUserMedia, delegar a la grabadora nativa
    if (!hasLiveAudioSupport()) {
        if (mobileMicInput) mobileMicInput.click();
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            currentRecordedBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioUrl = URL.createObjectURL(currentRecordedBlob);
            sampleAudioPreview.src = audioUrl;
            audioPreviewContainer.classList.remove("hidden");
        };

        mediaRecorder.start();
        recordBtn.classList.add("recording");
        recordIcon.className = "fa-solid fa-stop";
        recorderHint.innerText = "Grabando... Habla claro (3 - 10 seg). Haz clic para detener.";

        recordInterval = setInterval(() => {
            recordSeconds++;
            const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
            const secs = String(recordSeconds % 60).padStart(2, '0');
            recordingTimer.innerText = `${mins}:${secs}`;
        }, 1000);

    } catch (err) {
        console.warn("Error en getUserMedia:", err);
        showToast("No se pudo iniciar grabación en navegador. Abriendo grabadora del dispositivo...", "info");
        if (mobileMicInput) mobileMicInput.click();
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }

    clearInterval(recordInterval);
    recordBtn.classList.remove("recording");
    recordIcon.className = "fa-solid fa-microphone";
    recorderHint.innerText = "Grabación lista. Revisa la vista previa y guárdala.";
}

async function uploadRecordedSample() {
    if (!currentRecordedBlob) {
        showToast("No hay grabación activa para guardar.", "error");
        return;
    }

    const customName = sampleNameInput.value.trim() || `Grabacion_${Date.now()}`;

    // Determinar la extensión correcta según el archivo o tipo MIME
    let fileExt = "wav";
    if (currentRecordedBlob.name && currentRecordedBlob.name.includes(".")) {
        fileExt = currentRecordedBlob.name.split('.').pop().toLowerCase();
    } else if (currentRecordedBlob.type) {
        const type = currentRecordedBlob.type.toLowerCase();
        if (type.includes("mp4") || type.includes("m4a")) fileExt = "m4a";
        else if (type.includes("aac")) fileExt = "aac";
        else if (type.includes("mp3") || type.includes("mpeg")) fileExt = "mp3";
        else if (type.includes("ogg")) fileExt = "ogg";
        else if (type.includes("webm")) fileExt = "webm";
        else if (type.includes("3gp")) fileExt = "3gp";
    }

    const formData = new FormData();
    formData.append("file", currentRecordedBlob, `${customName}.${fileExt}`);
    formData.append("name", customName);

    try {
        saveSampleBtn.disabled = true;
        saveSampleBtn.innerText = "Guardando...";

        const res = await fetch(`${API_BASE}/api/samples/upload`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Error subiendo muestra.");
        }
        const data = await res.json();

        showToast(data.message, "success");
        audioPreviewContainer.classList.add("hidden");
        sampleNameInput.value = "";
        currentRecordedBlob = null;
        
        loadSamples();
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        saveSampleBtn.disabled = false;
        saveSampleBtn.innerHTML = `<i class="fa-solid fa-floppy-disk"></i> Guardar Muestra`;
    }
}

// File Dropzone logic
function initDropzone() {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--primary)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-color)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        showToast("Subiendo archivo de audio...", "info");
        const res = await fetch(`${API_BASE}/api/samples/upload`, {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("Error procesando audio");
        const data = await res.json();
        showToast(data.message, "success");
        loadSamples();
    } catch (err) {
        showToast(err.message, "error");
    }
}

// Load Samples for Selector & Grid
async function loadSamples() {
    try {
        const res = await fetch(`${API_BASE}/api/samples`);
        if (!res.ok) return;
        const samples = await res.json();

        // Update Dropdown in Generator
        selectSample.innerHTML = "";
        if (samples.length === 0) {
            selectSample.innerHTML = `<option value="">-- Graba o sube una muestra primero --</option>`;
        } else {
            samples.forEach(s => {
                const opt = document.createElement("option");
                opt.value = s.filename;
                opt.innerText = `🎙️ ${s.filename} (${s.duration_seconds}s)`;
                selectSample.appendChild(opt);
            });
        }

        // Render Samples Grid Tab
        samplesGrid.innerHTML = "";
        if (samples.length === 0) {
            samplesGrid.innerHTML = `<p class="subtitle">No hay muestras de voz guardadas.</p>`;
            return;
        }

        samples.forEach(s => {
            const card = document.createElement("div");
            card.className = "audio-card";
            card.innerHTML = `
                <div class="audio-card-title">
                    <span><i class="fa-solid fa-microphone"></i> ${s.filename}</span>
                    <button type="button" class="btn btn-danger btn-sm" onclick="deleteSample('${s.filename}')">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
                <div class="audio-card-meta">
                    Duración: ${s.duration_seconds} seg | Tamaño: ${(s.size_bytes / 1024).toFixed(1)} KB
                </div>
                <audio controls src="${API_BASE}/api/clips/download/${s.filename}"></audio>
            `;
            samplesGrid.appendChild(card);
        });

    } catch (err) {
        console.error("Error cargando muestras:", err);
    }
}

async function deleteSample(filename) {
    if (!confirm(`¿Eliminar la muestra '${filename}'?`)) return;
    try {
        const res = await fetch(`${API_BASE}/api/samples/${filename}`, { method: "DELETE" });
        if (res.ok) {
            showToast(`Muestra '${filename}' eliminada.`, "info");
            loadSamples();
        }
    } catch (err) {
        showToast("Error al eliminar.", "error");
    }
}

// ==========================================================================
// Settings Persistence Management (localStorage + backend/data/settings.json)
// ==========================================================================
const SETTINGS_STORAGE_KEY = "voicecloner_user_settings";

function getCurrentFormSettings() {
    return {
        language: document.getElementById("selectLang")?.value || "es",
        output_format: document.getElementById("selectFormat")?.value || "wav",
        temperature: parseFloat(document.getElementById("tempInput")?.value || "0.75"),
        top_p: parseFloat(document.getElementById("topPInput")?.value || "0.85"),
        top_k: parseInt(document.getElementById("topKInput")?.value || "50", 10),
        repetition_penalty: parseFloat(document.getElementById("repPenaltyInput")?.value || "3.5"),
        length_penalty: parseFloat(document.getElementById("lengthPenaltyInput")?.value || "1.0"),
        speed: parseFloat(document.getElementById("speedInput")?.value || "1.0"),
        use_rvc: !!document.getElementById("enableRvcCheckbox")?.checked,
        rvc_model: document.getElementById("selectRvcModel")?.value || "david",
        pitch_shift: parseInt(document.getElementById("rvcPitchInput")?.value || "0", 10),
        evaluate_quality: !!document.getElementById("evalQualityCheckbox")?.checked,
        auto_retry: !!document.getElementById("autoRetryCheckbox")?.checked,
        min_quality_threshold: parseInt(document.getElementById("qualityThresholdInput")?.value || "75", 10),
        max_retry_attempts: parseInt(document.getElementById("maxRetriesSelect")?.value || "2", 10)
    };
}

let saveSettingsDebounce = null;
function saveSettings(notifyUser = false) {
    const settings = getCurrentFormSettings();
    try {
        localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch (e) {
        console.warn("Could not save to localStorage", e);
    }

    const statusText = document.getElementById("settingsStatusText");
    if (statusText) {
        statusText.innerHTML = `<span style="color: #6ee7b7;"><i class="fa-solid fa-check"></i> Guardando cambios...</span>`;
    }

    if (saveSettingsDebounce) clearTimeout(saveSettingsDebounce);
    saveSettingsDebounce = setTimeout(async () => {
        try {
            await fetch(`${API_BASE}/api/settings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(settings)
            });
            if (statusText) {
                statusText.innerHTML = `<span style="color: #34d399;"><i class="fa-solid fa-check-double"></i> Parámetros sincronizados y guardados</span>`;
                setTimeout(() => {
                    if (statusText) statusText.innerText = "Parámetros sincronizados y guardados para futuras sesiones";
                }, 3500);
            }
            if (notifyUser) showToast("Ajustes guardados para futuras sesiones.", "success");
        } catch (e) {
            console.warn("Could not persist to backend settings.json", e);
            if (statusText) statusText.innerText = "Guardado localmente en este navegador";
        }
    }, 600);
}

async function loadSettings() {
    let settings = null;
    try {
        const local = localStorage.getItem(SETTINGS_STORAGE_KEY);
        if (local) settings = JSON.parse(local);
    } catch (e) {
        console.warn("Could not read localStorage", e);
    }

    if (!settings) {
        try {
            const res = await fetch(`${API_BASE}/api/settings`);
            if (res.ok) settings = await res.json();
        } catch (e) {
            console.warn("Could not fetch backend settings", e);
        }
    }

    if (settings) {
        applySettingsToForm(settings);
    }
}

function applySettingsToForm(s) {
    if (!s) return;
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined && val !== null) el.value = val;
    };
    const setChecked = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined && val !== null) el.checked = !!val;
    };

    if (s.language) setVal("selectLang", s.language);
    if (s.output_format) setVal("selectFormat", s.output_format);
    if (s.temperature !== undefined) setVal("tempInput", s.temperature);
    if (s.top_p !== undefined) setVal("topPInput", s.top_p);
    if (s.top_k !== undefined) setVal("topKInput", s.top_k);
    if (s.repetition_penalty !== undefined) setVal("repPenaltyInput", s.repetition_penalty);
    if (s.length_penalty !== undefined) setVal("lengthPenaltyInput", s.length_penalty);
    if (s.speed !== undefined) setVal("speedInput", s.speed);

    if (s.use_rvc !== undefined) setChecked("enableRvcCheckbox", s.use_rvc);
    if (s.rvc_model) setVal("selectRvcModel", s.rvc_model);
    if (s.pitch_shift !== undefined) setVal("rvcPitchInput", s.pitch_shift);

    if (s.evaluate_quality !== undefined) setChecked("evalQualityCheckbox", s.evaluate_quality);
    if (s.auto_retry !== undefined) setChecked("autoRetryCheckbox", s.auto_retry);
    if (s.min_quality_threshold !== undefined) setVal("qualityThresholdInput", s.min_quality_threshold);
    if (s.max_retry_attempts !== undefined) setVal("maxRetriesSelect", s.max_retry_attempts);

    // Actualizar etiquetas numéricas
    const tempVal = document.getElementById("tempVal");
    if (tempVal && s.temperature !== undefined) tempVal.innerText = parseFloat(s.temperature).toFixed(2);
    const topPVal = document.getElementById("topPVal");
    if (topPVal && s.top_p !== undefined) topPVal.innerText = parseFloat(s.top_p).toFixed(2);
    const topKVal = document.getElementById("topKVal");
    if (topKVal && s.top_k !== undefined) topKVal.innerText = parseInt(s.top_k, 10);
    const repPenaltyVal = document.getElementById("repPenaltyVal");
    if (repPenaltyVal && s.repetition_penalty !== undefined) repPenaltyVal.innerText = parseFloat(s.repetition_penalty).toFixed(2);
    const lengthPenaltyVal = document.getElementById("lengthPenaltyVal");
    if (lengthPenaltyVal && s.length_penalty !== undefined) lengthPenaltyVal.innerText = parseFloat(s.length_penalty).toFixed(2);
    const speedVal = document.getElementById("speedVal");
    if (speedVal && s.speed !== undefined) speedVal.innerText = `${parseFloat(s.speed).toFixed(2)}x`;
    const rvcPitchVal = document.getElementById("rvcPitchVal");
    if (rvcPitchVal && s.pitch_shift !== undefined) rvcPitchVal.innerText = parseInt(s.pitch_shift, 10);

    const rvcContainer = document.getElementById("rvcOptionsContainer");
    if (rvcContainer) rvcContainer.style.display = s.use_rvc ? "grid" : "none";

    const qualityThresholdVal = document.getElementById("qualityThresholdVal");
    if (qualityThresholdVal && s.min_quality_threshold !== undefined) qualityThresholdVal.innerText = `${s.min_quality_threshold}%`;

    const retrySettingsRow = document.getElementById("retrySettingsRow");
    if (retrySettingsRow) retrySettingsRow.style.display = s.auto_retry ? "grid" : "none";
}

function resetDefaultSettings() {
    const defaults = {
        language: "es",
        output_format: "wav",
        temperature: 0.75,
        top_p: 0.85,
        top_k: 50,
        repetition_penalty: 3.5,
        length_penalty: 1.0,
        speed: 1.0,
        use_rvc: false,
        rvc_model: "david",
        pitch_shift: 0,
        evaluate_quality: true,
        auto_retry: false,
        min_quality_threshold: 75,
        max_retry_attempts: 2
    };
    applySettingsToForm(defaults);
    saveSettings(true);
    showToast("Parámetros restablecidos a valores recomendados.", "info");
}

// ==========================================================================
// Generator Logic
// ==========================================================================
function initGenerator() {
    generateClipBtn.addEventListener("click", generateClip);

    function setupSlider(inputId, valId, formatFn) {
        const input = document.getElementById(inputId);
        const val = document.getElementById(valId);
        if (!input || !val) return;
        const update = () => {
            val.innerText = formatFn(input.value);
            saveSettings();
        };
        input.addEventListener("input", update);
        input.addEventListener("change", update);
        val.innerText = formatFn(input.value);
    }

    setupSlider("tempInput", "tempVal", v => parseFloat(v).toFixed(2));
    setupSlider("topPInput", "topPVal", v => parseFloat(v).toFixed(2));
    setupSlider("topKInput", "topKVal", v => parseInt(v, 10));
    setupSlider("repPenaltyInput", "repPenaltyVal", v => parseFloat(v).toFixed(2));
    setupSlider("lengthPenaltyInput", "lengthPenaltyVal", v => parseFloat(v).toFixed(2));
    setupSlider("speedInput", "speedVal", v => `${parseFloat(v).toFixed(2)}x`);
    setupSlider("rvcPitchInput", "rvcPitchVal", v => parseInt(v, 10));
    setupSlider("qualityThresholdInput", "qualityThresholdVal", v => `${parseInt(v, 10)}%`);

    // RVC controls
    const enableRvcCheckbox = document.getElementById("enableRvcCheckbox");
    const rvcOptionsContainer = document.getElementById("rvcOptionsContainer");
    if (enableRvcCheckbox && rvcOptionsContainer) {
        enableRvcCheckbox.addEventListener("change", (e) => {
            rvcOptionsContainer.style.display = e.target.checked ? "grid" : "none";
            saveSettings();
        });
    }

    const selectRvcModel = document.getElementById("selectRvcModel");
    if (selectRvcModel) {
        selectRvcModel.addEventListener("change", () => saveSettings());
    }

    // Quality & Auto-retry Controls
    const evalQualityCheckbox = document.getElementById("evalQualityCheckbox");
    const autoRetryCheckbox = document.getElementById("autoRetryCheckbox");
    const retrySettingsRow = document.getElementById("retrySettingsRow");
    const maxRetriesSelect = document.getElementById("maxRetriesSelect");

    if (evalQualityCheckbox) {
        evalQualityCheckbox.addEventListener("change", (e) => {
            saveSettings();
        });
    }

    if (autoRetryCheckbox && retrySettingsRow) {
        autoRetryCheckbox.addEventListener("change", (e) => {
            retrySettingsRow.style.display = e.target.checked ? "grid" : "none";
            saveSettings();
        });
    }

    if (maxRetriesSelect) {
        maxRetriesSelect.addEventListener("change", () => saveSettings());
    }

    // Language & Format selectors
    if (selectLang) selectLang.addEventListener("change", () => saveSettings());
    const selectFormat = document.getElementById("selectFormat");
    if (selectFormat) selectFormat.addEventListener("change", () => saveSettings());

    // Reset settings button
    const resetSettingsBtn = document.getElementById("resetSettingsBtn");
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener("click", resetDefaultSettings);
    }

    loadRvcModels();
}

function updateRvcModelsDropdown(models) {
    const selectRvcModel = document.getElementById("selectRvcModel");
    if (!selectRvcModel || !models) return;
    const currentVal = selectRvcModel.value;
    if (models.length === 0) {
        selectRvcModel.innerHTML = `<option value="">-- No hay modelos RVC (.pth) --</option>`;
        return;
    }
    selectRvcModel.innerHTML = models.map(m => `<option value="${m}">${m}.pth</option>`).join("");
    if (currentVal && models.includes(currentVal)) {
        selectRvcModel.value = currentVal;
    }
}

async function loadRvcModels() {
    try {
        const res = await fetch(`${API_BASE}/api/rvc/models`);
        if (!res.ok) return;
        const data = await res.json();
        updateRvcModelsDropdown(data.models);
        // Volver a aplicar el modelo RVC guardado si coincide
        const local = localStorage.getItem(SETTINGS_STORAGE_KEY);
        if (local) {
            const s = JSON.parse(local);
            if (s.rvc_model && data.models.includes(s.rvc_model)) {
                const selectRvcModel = document.getElementById("selectRvcModel");
                if (selectRvcModel) selectRvcModel.value = s.rvc_model;
            }
        }
    } catch (e) {
        console.warn("Could not load RVC models", e);
    }
}

let progressTimer = null;
let currentProgress = 0;

function startGenerationProgress(useRvc, text, autoRetry, evaluateQuality) {
    const fill = document.getElementById("progressBarFill");
    const percent = document.getElementById("progressPercent");
    const statusDetail = document.getElementById("loaderStatusDetail");
    const stepIndicator = document.getElementById("loaderStepIndicator");
    const subtext = document.getElementById("loaderSubtext");

    if (progressTimer) clearInterval(progressTimer);
    currentProgress = 2;
    if (fill) fill.style.width = "2%";
    if (percent) percent.innerText = "0%";
    if (statusDetail) statusDetail.innerText = "Inicializando redes neuronales...";
    if (stepIndicator) stepIndicator.innerText = useRvc ? "Paso 1 de 2" : "Paso 1 de 1";
    if (subtext) subtext.innerText = autoRetry ? "Modo con autoreintento activo (QA inteligente)" : "Procesando inferencia de red neuronal en ./backend/models/";

    const wordCount = Math.max(text.trim().split(/\s+/).length, 3);
    const estimatedSecs = (wordCount * 0.75) + (useRvc ? 13 : 4) + (evaluateQuality ? 1.5 : 0);
    const intervalMs = 200;
    const totalTicks = (estimatedSecs * 1000) / intervalMs;
    const baseIncrement = 92 / Math.max(totalTicks, 10);

    progressTimer = setInterval(() => {
        if (currentProgress < 95) {
            const factor = currentProgress > 85 ? 0.25 : (currentProgress > 65 ? 0.6 : 1.0);
            currentProgress += baseIncrement * factor;
            if (currentProgress > 95) currentProgress = 95;

            const val = Math.round(currentProgress);
            if (fill) fill.style.width = `${val}%`;
            if (percent) percent.innerText = `${val}%`;

            if (useRvc) {
                if (val < 42) {
                    if (statusDetail) statusDetail.innerText = "Generando audio base con XTTS-v2...";
                    if (stepIndicator) stepIndicator.innerText = "Paso 1 de 2: Síntesis";
                } else if (val < 78) {
                    if (statusDetail) statusDetail.innerText = "Extrayendo entonación y armónicos (RMVPE & ContentVec)...";
                    if (stepIndicator) stepIndicator.innerText = "Paso 2 de 2: RVC";
                } else if (val < 90) {
                    if (statusDetail) statusDetail.innerText = "Afinando timbre hiperrealista con modelo RVC...";
                    if (stepIndicator) stepIndicator.innerText = "Paso 2 de 2: Afinación";
                } else {
                    if (statusDetail) statusDetail.innerText = evaluateQuality ? "Evaluando calidad de audio y timbre por IA..." : "Codificando archivo final...";
                }
            } else {
                if (val < 65) {
                    if (statusDetail) statusDetail.innerText = "Sintetizando fonemas con XTTS-v2...";
                } else if (val < 88) {
                    if (statusDetail) statusDetail.innerText = "Codificando audio final...";
                } else {
                    if (statusDetail) statusDetail.innerText = evaluateQuality ? "Evaluando calidad y naturalidad por IA..." : "Completando archivo...";
                }
                if (stepIndicator) stepIndicator.innerText = "Paso 1 de 1";
            }
        }
    }, intervalMs);
}

function completeGenerationProgress() {
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
    const fill = document.getElementById("progressBarFill");
    const percent = document.getElementById("progressPercent");
    const statusDetail = document.getElementById("loaderStatusDetail");
    const stepIndicator = document.getElementById("loaderStepIndicator");

    currentProgress = 100;
    if (fill) fill.style.width = "100%";
    if (percent) percent.innerText = "100%";
    if (statusDetail) statusDetail.innerText = "¡Audio generado y evaluado con éxito!";
    if (stepIndicator) stepIndicator.innerText = "¡Listo!";
}

function stopGenerationProgress() {
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
}

function renderQualityScorecard(q) {
    const scorecard = document.getElementById("qualityScorecard");
    if (!scorecard) return;

    if (!q || q.overall_score === undefined) {
        scorecard.classList.add("hidden");
        return;
    }

    const overallPill = document.getElementById("scoreOverallPill");
    const overallVal = document.getElementById("scoreOverallVal");
    const ratingVal = document.getElementById("scoreRatingVal");
    const detailsText = document.getElementById("scoreDetailsText");
    const attemptsNotice = document.getElementById("scoreAttemptsNotice");
    const attemptsText = document.getElementById("scoreAttemptsText");

    const simVal = document.getElementById("metricSimilarityVal");
    const simFill = document.getElementById("metricSimilarityFill");
    const clarVal = document.getElementById("metricClarityVal");
    const clarFill = document.getElementById("metricClarityFill");
    const pacVal = document.getElementById("metricPacingVal");
    const pacFill = document.getElementById("metricPacingFill");

    // Puntaje Global
    const score = Math.round(q.overall_score);
    if (overallVal) overallVal.innerText = `${score}%`;
    if (ratingVal) ratingVal.innerText = q.rating || "Normal";

    if (overallPill) {
        overallPill.className = "score-pill";
        if (score >= 80) {
            overallPill.classList.add("score-excellent");
        } else if (score >= 65) {
            overallPill.classList.add("score-good");
        } else {
            overallPill.classList.add("score-fair");
        }
    }

    if (detailsText) {
        detailsText.innerText = q.details || "Evaluación acústica completada exitosamente.";
    }

    // Historial de intentos
    if (attemptsNotice && attemptsText) {
        const count = q.attempts_count || q.attempt_number || 1;
        const total = q.total_attempts || 1;
        if (count > 1 || q.threshold_set) {
            const metText = q.met_threshold ? "Alcanzó el umbral deseado" : "Mejor resultado alcanzado";
            attemptsText.innerText = `${count} intento(s) realizado(s) • ${metText} (${q.threshold_set || 75}%)`;
            attemptsNotice.classList.remove("hidden");
        } else {
            attemptsNotice.classList.add("hidden");
        }
    }

    // Métricas individuales
    const sim = Math.round(q.speaker_similarity || 0);
    if (simVal) simVal.innerText = `${sim}%`;
    if (simFill) simFill.style.width = `${sim}%`;

    const clar = Math.round(q.acoustic_clarity || 0);
    const mos = q.mos_score !== undefined ? q.mos_score : (1.0 + clar * 0.038).toFixed(1);
    if (clarVal) clarVal.innerText = `${mos} / 5.0 ⭐ (${clar}%)`;
    if (clarFill) clarFill.style.width = `${clar}%`;

    const pac = Math.round(q.pacing_score || 0);
    if (pacVal) pacVal.innerText = `${pac}%`;
    if (pacFill) pacFill.style.width = `${pac}%`;

    scorecard.classList.remove("hidden");
}

async function generateClip() {
    const selectedFile = selectSample.value;
    const text = textPrompt.value.trim();
    const lang = selectLang.value;
    const tempInput = document.getElementById("tempInput");
    const topPInput = document.getElementById("topPInput");
    const topKInput = document.getElementById("topKInput");
    const repPenaltyInput = document.getElementById("repPenaltyInput");
    const lengthPenaltyInput = document.getElementById("lengthPenaltyInput");
    const speedInput = document.getElementById("speedInput");
    const selectFormat = document.getElementById("selectFormat");
    const outputFormat = selectFormat ? selectFormat.value : "wav";

    const temperature = tempInput ? parseFloat(tempInput.value) : 0.75;
    const topP = topPInput ? parseFloat(topPInput.value) : 0.85;
    const topK = topKInput ? parseInt(topKInput.value, 10) : 50;
    const repetitionPenalty = repPenaltyInput ? parseFloat(repPenaltyInput.value) : 3.5;
    const lengthPenalty = lengthPenaltyInput ? parseFloat(lengthPenaltyInput.value) : 1.0;
    const speed = speedInput ? parseFloat(speedInput.value) : 1.0;

    const enableRvcCheckbox = document.getElementById("enableRvcCheckbox");
    const selectRvcModel = document.getElementById("selectRvcModel");
    const rvcPitchInput = document.getElementById("rvcPitchInput");

    const useRvc = enableRvcCheckbox ? enableRvcCheckbox.checked : false;
    const rvcModel = selectRvcModel ? selectRvcModel.value : null;
    const pitchShift = rvcPitchInput ? parseInt(rvcPitchInput.value, 10) : 0;

    // Quality & Auto-retry inputs
    const evalQualityCheckbox = document.getElementById("evalQualityCheckbox");
    const autoRetryCheckbox = document.getElementById("autoRetryCheckbox");
    const qualityThresholdInput = document.getElementById("qualityThresholdInput");
    const maxRetriesSelect = document.getElementById("maxRetriesSelect");

    const evaluateQuality = evalQualityCheckbox ? evalQualityCheckbox.checked : true;
    const autoRetry = autoRetryCheckbox ? autoRetryCheckbox.checked : false;
    const minQualityThreshold = qualityThresholdInput ? parseInt(qualityThresholdInput.value, 10) : 75;
    const maxRetryAttempts = maxRetriesSelect ? parseInt(maxRetriesSelect.value, 10) : 2;

    if (!selectedFile && !useRvc) {
        showToast("Por favor selecciona una muestra de voz de referencia.", "error");
        return;
    }
    if (!text) {
        showToast("Por favor ingresa un texto para sintetizar.", "error");
        return;
    }

    try {
        generateClipBtn.disabled = true;
        clipLoader.classList.remove("hidden");
        resultCard.classList.add("hidden");
        startGenerationProgress(useRvc, text, autoRetry, evaluateQuality);

        const res = await fetch(`${API_BASE}/api/clone/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                sample_filename: selectedFile,
                language: lang,
                output_format: outputFormat,
                temperature: temperature,
                top_p: topP,
                top_k: topK,
                repetition_penalty: repetitionPenalty,
                length_penalty: lengthPenalty,
                speed: speed,
                use_rvc: useRvc,
                rvc_model: rvcModel,
                pitch_shift: pitchShift,
                evaluate_quality: evaluateQuality,
                auto_retry: autoRetry,
                min_quality_threshold: minQualityThreshold,
                max_retry_attempts: maxRetryAttempts
            })
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Error en la generación de clip");
        }

        const data = await res.json();
        completeGenerationProgress();

        // Pausa breve para mostrar el 100% completado al usuario
        await new Promise(r => setTimeout(r, 450));

        showToast("¡Clip de voz generado exitosamente!", "success");

        // Set Result Player
        const audioUrl = `${API_BASE}${data.download_url}?t=${Date.now()}`;
        resultAudioPlayer.src = audioUrl;
        downloadResultBtn.href = audioUrl;
        const ext = data.clip_filename.split('.').pop().toUpperCase();
        downloadResultBtn.innerHTML = `<i class="fa-solid fa-download"></i> Descargar ${ext}`;

        // Render Quality Scorecard si existe
        renderQualityScorecard(data.quality);

        resultCard.classList.remove("hidden");

        loadClips();

    } catch (err) {
        stopGenerationProgress();
        showToast(err.message, "error");
    } finally {
        stopGenerationProgress();
        generateClipBtn.disabled = false;
        clipLoader.classList.add("hidden");
    }
}

// Load Clips Grid
async function loadClips() {
    try {
        const res = await fetch(`${API_BASE}/api/clips`);
        if (!res.ok) return;
        const clips = await res.json();

        clipsGrid.innerHTML = "";
        if (clips.length === 0) {
            clipsGrid.innerHTML = `<p class="subtitle">Aún no se han generado clips de audio.</p>`;
            return;
        }

        clips.forEach(c => {
            const card = document.createElement("div");
            card.className = "audio-card";

            // Badge de calidad si existe
            let qualityBadge = "";
            if (c.quality && c.quality.overall_score !== undefined) {
                const score = Math.round(c.quality.overall_score);
                const cls = score >= 80 ? "quality-high" : (score >= 65 ? "quality-mid" : "quality-low");
                qualityBadge = `<span class="badge-clip-quality ${cls}" title="Calificación IA: ${c.quality.rating || ''} (${score}%)"><i class="fa-solid fa-star"></i> ${score}%</span>`;
            }

            card.innerHTML = `
                <div class="audio-card-title">
                    <span><i class="fa-solid fa-music"></i> ${c.filename} ${qualityBadge}</span>
                    <button type="button" class="btn btn-danger btn-sm" onclick="deleteClip('${c.filename}')">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
                <div class="audio-card-meta">
                    Duración: ${c.duration_seconds} seg | Generado: ${new Date(c.modified_time * 1000).toLocaleTimeString()}
                </div>
                <audio controls src="${API_BASE}${c.download_url}"></audio>
                <a href="${API_BASE}${c.download_url}" download class="btn btn-outline btn-block mt-10">
                    <i class="fa-solid fa-download"></i> Descargar
                </a>
            `;
            clipsGrid.appendChild(card);
        });

    } catch (err) {
        console.error("Error cargando clips:", err);
    }
}

async function deleteClip(filename) {
    if (!confirm(`¿Eliminar el clip '${filename}'?`)) return;
    try {
        const res = await fetch(`${API_BASE}/api/clips/${filename}`, { method: "DELETE" });
        if (res.ok) {
            showToast(`Clip '${filename}' eliminado.`, "info");
            loadClips();
        }
    } catch (err) {
        showToast("Error al eliminar el clip.", "error");
    }
}
