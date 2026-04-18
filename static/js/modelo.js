// ── TABS ─────────────────────────────────────────────────
function cambiarTab(nombre, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + nombre).classList.add('active');
}

// ── NOMBRE CSV ENTRENAMIENTO ──────────────────────────────
function mostrarNombreCSV() {
    const f = document.getElementById('inputCSV').files[0];
    document.getElementById('nombreCSV').textContent =
        f ? f.name : 'Seleccionar CSV de características';
}

// ── LOG ───────────────────────────────────────────────────
function log(msg, tipo = '') {
    const el = document.getElementById('estadoLog');
    el.innerHTML += `<div class="log-${tipo}">${msg}</div>`;
    el.scrollTop = el.scrollHeight;
}

function clearLog() {
    document.getElementById('estadoLog').innerHTML = '';
}

// ── ENTRENAMIENTO ─────────────────────────────────────────
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
    form.append('archivo',    archivo);
    form.append('features',   features.join(','));
    form.append('epochs',     epochs);
    form.append('batch',      batch);
    form.append('lr',         lr);
    form.append('split',      split);
    form.append('reentrenar', reentrenar ? '1' : '0');

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

// ── GUARDAR MODELO ────────────────────────────────────────
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

// ── ANÁLISIS ──────────────────────────────────────────────
async function cargarAnalisis() {
    const archivo = document.getElementById('inputCSVAnalisis').files[0];
    if (!archivo) { alert('Selecciona un CSV'); return; }

    const eliminarOutliers = document.getElementById('chkOutliers').checked;

    const form = new FormData();
    form.append('archivo',           archivo);
    form.append('eliminar_outliers', eliminarOutliers ? '1' : '0');

    // Ocultar resultados anteriores
    document.getElementById('emptyAnalisis').style.display  = 'none';
    document.getElementById('graficasWrap').style.display   = 'none';
    document.getElementById('cardsClases').innerHTML        = '';
    document.getElementById('labelClases').style.display    = 'none';
    document.getElementById('outliersBanner').style.display = 'none';

    try {
        const r = await fetch('/modelo/analisis', { method: 'POST', body: form });
        const d = await r.json();

        if (d.error) { alert(d.error); return; }

        // ── Banner outliers ──────────────────────────────────
        if (d.outliers_info) {
            const info = d.outliers_info;
            document.getElementById('outliersTexto').textContent  =
                `Outliers eliminados: ${info.eliminados}`;
            document.getElementById('outliersDetalle').textContent =
                `${info.antes} muestras originales → ${info.despues} muestras limpias`;
            document.getElementById('outliersBanner').style.display = 'flex';
        }

        // ── Cards de clases ──────────────────────────────────
        const wrap = document.getElementById('cardsClases');
        const COLORES = ['#5b8fb9','#ef4444','#16a34a','#f59e0b',
                         '#8b5cf6','#ec4899','#14b8a6','#f97316'];

        Object.entries(d.clases).forEach(([clase, info], i) => {
            const col  = COLORES[i % COLORES.length];
            const card = document.createElement('div');
            card.className = 'clase-card';
            card.style.borderTop = `4px solid ${col}`;
            card.innerHTML = `
                <div class="clase-nombre">${clase}</div>
                <div class="clase-count">${info.count}</div>
                <div class="clase-pct">${info.pct}% del total</div>
                <div class="clase-bar-wrap">
                    <div class="clase-bar" style="width:${info.pct}%; background:${col}"></div>
                </div>`;
            wrap.appendChild(card);
        });

        // Card total
        const total = document.createElement('div');
        total.className = 'clase-card total-card';
        total.innerHTML = `
            <div class="clase-nombre">Total</div>
            <div class="clase-count">${d.total}</div>
            <div class="clase-pct">muestras</div>`;
        wrap.appendChild(total);

        document.getElementById('labelClases').style.display = 'flex';

        // ── Gráficas ─────────────────────────────────────────
        document.getElementById('imgBalance').src  = `data:image/png;base64,${d.img_balance}`;
        document.getElementById('imgPairplot').src = `data:image/png;base64,${d.img_pairplot}`;
        document.getElementById('graficasWrap').style.display = 'block';

    } catch(e) {
        alert('Error al cargar análisis: ' + e);
        document.getElementById('emptyAnalisis').style.display = 'flex';
    }
}