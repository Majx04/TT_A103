// ── TABS ────────────────────────────────────────────────
function cambiarTab(nombre, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + nombre).classList.add('active');
}

// ── NOMBRE CSV ───────────────────────────────────────────
function mostrarNombreCSV() {
    const f = document.getElementById('inputCSV').files[0];
    document.getElementById('nombreCSV').textContent = f ? f.name : 'Seleccionar CSV de características';
}

// ── LOG HELPER ───────────────────────────────────────────
function log(msg, tipo = '') {
    const el = document.getElementById('estadoLog');
    el.innerHTML += `<div class="log-${tipo}">${msg}</div>`;
    el.scrollTop = el.scrollHeight;
}

function clearLog() {
    document.getElementById('estadoLog').innerHTML = '';
}

// ── ENTRENAMIENTO ────────────────────────────────────────
async function iniciarEntrenamiento(reentrenar) {
    const archivo = document.getElementById('inputCSV').files[0];
    if (!archivo) { alert('Selecciona un CSV primero'); return; }

    const features = [...document.querySelectorAll('.feat-check input:checked')]
        .map(cb => cb.value);
    if (!features.length) { alert('Selecciona al menos una característica'); return; }

    const epochs   = document.getElementById('epochs').value;
    const batch    = document.getElementById('batch').value;
    const lr       = document.getElementById('lr').value;
    const split    = document.getElementById('split').value;

    clearLog();
    log(`Modo: ${reentrenar ? 'Reentrenamiento' : 'Entrenamiento nuevo'}`);
    log(`Características: ${features.join(', ')}`);
    log(`Épocas: ${epochs} | Batch: ${batch} | LR: ${lr} | Split: ${split}`);
    log('Enviando datos al servidor...', 'muted');

    document.getElementById('progressWrap').style.display = 'block';
    document.getElementById('progressBar').style.width    = '10%';
    document.getElementById('cardMetricas').style.display = 'none';

    const form = new FormData();
    form.append('archivo',     archivo);
    form.append('features',    features.join(','));
    form.append('epochs',      epochs);
    form.append('batch',       batch);
    form.append('lr',          lr);
    form.append('split',       split);
    form.append('reentrenar',  reentrenar ? '1' : '0');

    try {
        document.getElementById('progressBar').style.width = '40%';
        const r = await fetch('/modelo/entrenar', { method: 'POST', body: form });
        document.getElementById('progressBar').style.width = '90%';
        const d = await r.json();

        if (d.error) {
            log(`Error: ${d.error}`, 'err');
            document.getElementById('progressBar').style.width = '0%';
            return;
        }

        document.getElementById('progressBar').style.width = '100%';
        log(`✔ Entrenamiento completado`, 'ok');
        log(`Precisión test: ${(d.test_accuracy * 100).toFixed(2)}%`, 'ok');
        log(`Mejor época: ${d.best_epoch} | Mejor val_acc: ${(d.best_val_accuracy * 100).toFixed(2)}%`, 'ok');

        // Mostrar métricas
        document.getElementById('mAcc').textContent    = `${(d.test_accuracy * 100).toFixed(1)}%`;
        document.getElementById('mValAcc').textContent = `${(d.best_val_accuracy * 100).toFixed(1)}%`;
        document.getElementById('mEpoch').textContent  = d.best_epoch;
        document.getElementById('mClases').textContent = d.classes.join(', ');
        document.getElementById('cardMetricas').style.display = 'block';

    } catch(e) {
        log(`Error de conexión: ${e}`, 'err');
        document.getElementById('progressBar').style.width = '0%';
    }
}

// ── GUARDAR MODELO ───────────────────────────────────────
async function guardarModelo() {
    try {
        const r = await fetch('/modelo/guardar', { method: 'POST' });
        const d = await r.json();
        if (d.ok) log(`✔ Modelo guardado como: ${d.nombre}`, 'ok');
        else      log(`Error al guardar: ${d.error}`, 'err');
    } catch(e) {
        log(`Error: ${e}`, 'err');
    }
}

// ── ANÁLISIS ─────────────────────────────────────────────
let datosAnalisis = null;
const COLORES = ['#5b8fb9','#ef4444','#16a34a','#f59e0b','#8b5cf6','#ec4899','#14b8a6'];

async function cargarAnalisis() {
    const archivo = document.getElementById('inputCSVAnalisis').files[0];
    if (!archivo) { alert('Selecciona un CSV'); return; }

    const form = new FormData();
    form.append('archivo', archivo);

    try {
        const r = await fetch('/modelo/analisis', { method: 'POST', body: form });
        datosAnalisis = await r.json();

        if (datosAnalisis.error) { alert(datosAnalisis.error); return; }

        dibujarClases();
        dibujarDispersion();
        dibujarStatsClases();
    } catch(e) {
        alert('Error al cargar datos: ' + e);
    }
}

function dibujarClases() {
    if (!datosAnalisis) return;
    const conteo  = datosAnalisis.conteo_clases;
    const clases  = Object.keys(conteo);
    const valores = Object.values(conteo);
    const max     = Math.max(...valores);

    const wrap = document.getElementById('grafClases');
    wrap.innerHTML = '';

    // Barras SVG simples
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${clases.length * 80 + 40} 200`);
    svg.style.width  = '100%';
    svg.style.height = '200px';

    clases.forEach((clase, i) => {
        const barH = (valores[i] / max) * 140;
        const x    = i * 80 + 30;
        const y    = 160 - barH;
        const col  = COLORES[i % COLORES.length];

        // Barra
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x); rect.setAttribute('y', y);
        rect.setAttribute('width', 50); rect.setAttribute('height', barH);
        rect.setAttribute('fill', col); rect.setAttribute('rx', 6);
        svg.appendChild(rect);

        // Valor
        const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        txt.setAttribute('x', x + 25); txt.setAttribute('y', y - 6);
        txt.setAttribute('text-anchor', 'middle');
        txt.setAttribute('font-size', '11'); txt.setAttribute('fill', '#334155');
        txt.textContent = valores[i];
        svg.appendChild(txt);

        // Etiqueta
        const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        lbl.setAttribute('x', x + 25); lbl.setAttribute('y', 178);
        lbl.setAttribute('text-anchor', 'middle');
        lbl.setAttribute('font-size', '11'); lbl.setAttribute('fill', '#64748b');
        lbl.textContent = clase;
        svg.appendChild(lbl);
    });

    wrap.appendChild(svg);
}

function dibujarDispersion() {
    if (!datosAnalisis) return;
    const ejeX  = document.getElementById('ejeX').value;
    const ejeY  = document.getElementById('ejeY').value;
    const puntos = datosAnalisis.puntos;
    const clases = [...new Set(puntos.map(p => p.label))];

    const canvas = document.getElementById('canvasDispersion');
    const ctx    = canvas.getContext('2d');
    canvas.width  = canvas.parentElement.clientWidth;
    canvas.height = 280;

    const W = canvas.width, H = canvas.height;
    const PAD = 40;

    const xs = puntos.map(p => p[ejeX]);
    const ys = puntos.map(p => p[ejeY]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);

    const scaleX = v => PAD + (v - xMin) / (xMax - xMin) * (W - PAD * 2);
    const scaleY = v => H - PAD - (v - yMin) / (yMax - yMin) * (H - PAD * 2);

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, W, H);

    // Ejes
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, H - PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - PAD, H - PAD); ctx.stroke();

    // Etiquetas ejes
    ctx.fillStyle = '#64748b'; ctx.font = '11px Segoe UI';
    ctx.textAlign = 'center';
    ctx.fillText(ejeX, W / 2, H - 6);
    ctx.save(); ctx.translate(12, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(ejeY, 0, 0); ctx.restore();

    // Puntos
    puntos.forEach(p => {
        const idx = clases.indexOf(p.label);
        ctx.fillStyle = COLORES[idx % COLORES.length] + 'cc';
        ctx.beginPath();
        ctx.arc(scaleX(p[ejeX]), scaleY(p[ejeY]), 4, 0, Math.PI * 2);
        ctx.fill();
    });

    // Leyenda
    clases.forEach((c, i) => {
        const lx = PAD + i * 90;
        ctx.fillStyle = COLORES[i % COLORES.length];
        ctx.fillRect(lx, 8, 12, 12);
        ctx.fillStyle = '#334155'; ctx.font = '11px Segoe UI'; ctx.textAlign = 'left';
        ctx.fillText(c, lx + 16, 19);
    });
}

function dibujarStatsClases() {
    if (!datosAnalisis) return;
    const wrap  = document.getElementById('statsClases');
    const stats = datosAnalisis.stats_clases;
    const clases = Object.keys(stats);

    wrap.innerHTML = '';

    clases.forEach((clase, i) => {
        const s   = stats[clase];
        const col = COLORES[i % COLORES.length];
        const card = document.createElement('div');
        card.className = 'clase-stat-card';
        card.innerHTML = `
            <div class="clase-stat-nombre">
                <div class="clase-stat-dot" style="background:${col}"></div>
                ${clase}
            </div>
            ${['WL','RMS','MAV','WAMP'].map(feat => `
            <div class="clase-stat-row">
                <span>${feat}</span>
                <span>μ ${s[feat]?.mean?.toFixed(4) ?? '—'} / σ ${s[feat]?.std?.toFixed(4) ?? '—'}</span>
            </div>`).join('')}
        `;
        wrap.appendChild(card);
    });
}