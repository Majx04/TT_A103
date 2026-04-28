// ── ESTADO ──────────────────────────────────────────────
let pausado   = false;
let chartData = new Array(5000).fill(0);

const PALETA = ["#00e5ff","#ff4081","#69f0ae","#ffea00","#ff6d00","#d500f9","#00b0ff","#76ff03"];
let clases = [];

// ── CANVAS ──────────────────────────────────────────────
const canvas = document.getElementById("emgCanvas");
const ctx    = canvas.getContext("2d");

function resizeCanvas() {
    // Fuerza dimensiones en píxeles reales, no CSS
    const wrapper = canvas.parentElement;
    canvas.width  = wrapper.clientWidth  || 800;
    canvas.height = wrapper.clientHeight || 300;
}

window.addEventListener("resize", () => { resizeCanvas(); drawChart(); });
resizeCanvas();

function drawChart() {
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Fondo
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);

    // Grid horizontal (5 líneas)
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth   = 0.8;
    for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * H;          // ← corregido: de arriba hacia abajo
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }

    if (!chartData.length) return;

    // ── Normalización dinámica ──────────────────────────
    // En lugar de asumir siempre 3.3 V de fondo, usamos el rango real
    // para que la señal sea visible aunque el ADC aún no tenga datos buenos
    const maxVal = Math.max(...chartData);
    const minVal = Math.min(...chartData);
    const range  = maxVal - minVal || 1;   // evita división por cero

    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth   = 1.2;
    ctx.beginPath();

    const step = W / (chartData.length - 1);
    for (let i = 0; i < chartData.length; i++) {
        const x = i * step;
        // Mapea [minVal, maxVal] → [H*0.95, H*0.05]  (margen de 5%)
        const y = H * 0.95 - ((chartData[i] - minVal) / range) * H * 0.9;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Etiqueta de escala (opcional, ayuda a depurar)
    ctx.fillStyle = "#556070";
    ctx.font      = "11px monospace";
    ctx.fillText(`max ${maxVal.toFixed(3)} V`, 6, 14);
    ctx.fillText(`min ${minVal.toFixed(3)} V`, 6, H - 6);
}

// ── FETCH DATA ──────────────────────────────────────────
async function fetchData() {
    if (pausado) return;
    try {
        const r = await fetch("/emg/data");
        if (!r.ok) return;
        const json = await r.json();

        // Validación: que sea array con números
        if (!Array.isArray(json) || json.length === 0) return;
        chartData = json;

        drawChart();

        // Stats
        const max = Math.max(...chartData);
        const min = Math.min(...chartData);
        const rms = Math.sqrt(chartData.reduce((s, v) => s + v * v, 0) / chartData.length);
        const mav = chartData.reduce((s, v) => s + Math.abs(v), 0) / chartData.length;

        document.getElementById("statMax").textContent = max.toFixed(3);
        document.getElementById("statMin").textContent = min.toFixed(3);
        document.getElementById("statRms").textContent = rms.toFixed(3);
        document.getElementById("statMav").textContent = mav.toFixed(3);

    } catch(e) {
        console.error("fetchData error:", e);
    }
}

// ── CLASIFICACIÓN ────────────────────────────────────────
async function fetchClasificacion() {
    try {
        const r = await fetch("/emg/clasificacion");
        if (!r.ok) return;
        const d = await r.json();

        const etiqueta  = d.etiqueta  || "—";
        const confianza = d.confianza || 0;

        document.getElementById("clfEtiqueta").textContent  = etiqueta;
        document.getElementById("clfConfianza").textContent = `Confianza: ${(confianza * 100).toFixed(1)}%`;
        document.getElementById("confBar").style.width      = `${(confianza * 100).toFixed(1)}%`;

        if (!clases.length) {
            try {
                const cr = await fetch("/emg/clases");
                clases = (await cr.json()).clases || [];
            } catch(e) {}
        }
        const idx   = clases.indexOf(etiqueta);
        const color = idx >= 0 ? PALETA[idx % PALETA.length] : "#00e5ff";
        document.getElementById("clfEtiqueta").style.color  = color;
        document.getElementById("confBar").style.background = color;

    } catch(e) {}
}

// ── PAUSA ────────────────────────────────────────────────
async function togglePausa() {
    try {
        const r = await fetch("/emg/toggle", { method: "POST" });
        const d = await r.json();
        pausado = d.pausado;

        const dot     = document.getElementById("estadoDot");
        const texto   = document.getElementById("estadoTexto");
        const icon    = document.getElementById("pausaIcon");
        const btnTxt  = document.getElementById("pausaTexto");
        const btnSave = document.getElementById("btnGuardar");

        if (pausado) {
            dot.classList.add("pausado");
            texto.textContent  = "PAUSADO";
            icon.className     = "fas fa-play";
            btnTxt.textContent = "Reanudar";
            btnSave.disabled   = false;
        } else {
            dot.classList.remove("pausado");
            texto.textContent  = "Adquisición activa";
            icon.className     = "fas fa-pause";
            btnTxt.textContent = "Pausar";
            btnSave.disabled   = true;
        }
    } catch(e) { showToast("Error al cambiar estado", "err"); }
}

// ── GUARDAR ──────────────────────────────────────────────
async function guardarMuestra() {
    const nombre = document.getElementById("nombreInput").value.trim();
    if (!nombre) { showToast("Escribe un nombre base", "err"); return; }

    const form = new FormData();
    form.append("nombre", nombre);

    try {
        const r = await fetch("/emg/guardar", { method: "POST", body: form });
        const d = await r.json();
        if (d.ok) showToast(`✔ Guardado: ${d.ok}`, "ok");
        else      showToast(`✖ ${d.error}`, "err");
    } catch(e) { showToast("Error al guardar", "err"); }
}

// ── TOAST ────────────────────────────────────────────────
function showToast(msg, type = "ok") {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className   = `show ${type}`;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.className = ""; }, 3200);
}

// ── LOOPS ────────────────────────────────────────────────
setInterval(fetchData,          50);
setInterval(fetchClasificacion, 500);

drawChart();  // primer frame