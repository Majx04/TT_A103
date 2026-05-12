// ════════════════════════════════════════════════
//  combinar_csv.js  →  static/js/combinar_csv.js
//  Usado por: templates/csv.html
// ════════════════════════════════════════════════

let pcFileObjects = [];

// ── Checkboxes del historial ──────────────────────────────
function onCb(cb) {
    cb.closest('.csv-item').classList.toggle('sel', cb.checked);
    updateCounter();
}

function selTodos(val) {
    document.querySelectorAll('.hist-cb').forEach(cb => {
        cb.checked = val;
        cb.closest('.csv-item').classList.toggle('sel', val);
    });
    updateCounter();
}

// ── Archivos desde PC ─────────────────────────────────────
const uploadLabel = document.querySelector('.boton-archivo-personalizado');

uploadLabel.addEventListener('dragover', e => {
    e.preventDefault();
    uploadLabel.style.borderColor = '#1a73e8';
    uploadLabel.style.background  = '#edf2ff';
});

uploadLabel.addEventListener('dragleave', () => {
    uploadLabel.style.borderColor = '';
    uploadLabel.style.background  = '';
});

uploadLabel.addEventListener('drop', e => {
    e.preventDefault();
    uploadLabel.style.borderColor = '';
    uploadLabel.style.background  = '';
    addFiles(Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.csv')));
});

function onFilesSelected(list) {
    addFiles(Array.from(list).filter(f => f.name.endsWith('.csv')));
    document.getElementById('file-upload').value = '';
}

function addFiles(files) {
    files.forEach(f => {
        if (!pcFileObjects.find(x => x.name === f.name)) pcFileObjects.push(f);
    });
    renderPcFiles();
    updateCounter();
}

function removeFile(i) {
    pcFileObjects.splice(i, 1);
    renderPcFiles();
    updateCounter();
}

function renderPcFiles() {
    const c = document.getElementById('pcFiles');
    c.innerHTML = '';
    pcFileObjects.forEach((f, i) => {
        const d = document.createElement('div');
        d.className = 'pc-file';
        d.innerHTML = `
            <i class="fas fa-file-csv" style="color:#22c55e;flex-shrink:0"></i>
            <span>${f.name}</span>
            <button onclick="removeFile(${i})" title="Quitar"><i class="fas fa-times"></i></button>`;
        c.appendChild(d);
    });
}

// ── Contador ──────────────────────────────────────────────
function updateCounter() {
    const n = document.querySelectorAll('.hist-cb:checked').length + pcFileObjects.length;
    document.getElementById('totalSel').textContent = n;
    document.getElementById('btnCombine').disabled  = n === 0;
}

// ── Combinar y descargar (en el navegador) ────────────────
async function combinar() {
    const btn      = document.getElementById('btnCombine');
    const rawName  = document.getElementById('outName').value.trim();
    const fileName = (rawName || 'combinado').replace(/[^a-zA-Z0-9_\-]/g, '_') + '.csv';
    const checked  = Array.from(document.querySelectorAll('.hist-cb:checked'));

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Combinando…';

    try {
        let header = null;
        let rows   = [];

        // 1. Leer CSVs del historial vía fetch
        for (const cb of checked) {
            const res = await fetch(`/static/emg/${encodeURIComponent(cb.value)}`);
            if (!res.ok) continue;
            const lines = (await res.text()).trim().split('\n').filter(l => l.trim());
            if (!lines.length) continue;
            if (!header) header = lines[0];
            rows = rows.concat(lines.slice(1));
        }

        // 2. Leer CSVs subidos desde PC
        for (const f of pcFileObjects) {
            const lines = (await f.text()).trim().split('\n').filter(l => l.trim());
            if (!lines.length) continue;
            if (!header) header = lines[0];
            rows = rows.concat(lines.slice(1));
        }

        if (!header || rows.length === 0) {
            showToast('No se pudo leer ningún CSV válido', 'err');
            return;
        }

        // 3. Construir blob y descargar
        const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv;charset=utf-8;' });
        const url  = URL.createObjectURL(blob);
        const a    = Object.assign(document.createElement('a'), { href: url, download: fileName });
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast(`✔ ${fileName} — ${rows.length} filas`, 'ok');

    } catch (e) {
        showToast('Error: ' + e.message, 'err');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-file-export"></i> Combinar y descargar';
    }
}

// ── Toast ─────────────────────────────────────────────────
function showToast(msg, type = 'ok') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className   = `show ${type}`;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.className = ''; }, 3500);
}