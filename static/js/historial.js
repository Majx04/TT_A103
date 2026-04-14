let filtroActivo = 'todos';

function filtrar() {
    const busqueda = document.getElementById('buscador').value.toLowerCase();
    const filas    = document.querySelectorAll('.fila-archivo');
    let visibles   = 0;

    filas.forEach(fila => {
        const tipo    = fila.getAttribute('data-tipo');
        const nombre  = fila.getAttribute('data-nombre').toLowerCase();
        const matchTipo   = filtroActivo === 'todos' || tipo === filtroActivo;
        const matchNombre = nombre.includes(busqueda);

        fila.style.display = (matchTipo && matchNombre) ? '' : 'none';
        if (matchTipo && matchNombre) visibles++;
    });

    const filaVacia = document.getElementById('filaVacia');
    if (filaVacia) filaVacia.style.display = visibles === 0 ? '' : 'none';

    document.getElementById('contadorVisible').textContent =
        `(${visibles} archivo${visibles !== 1 ? 's' : ''})`;
}
async function eliminar(nombre, btn) {
    if (!confirm(`¿Eliminar ${nombre}?`)) return;

    const form = new FormData();
    const r = await fetch(`/historial/eliminar/${nombre}`, { method: "POST", body: form });
    const d = await r.json();

    if (d.ok) {
        const fila = btn.closest('tr');
        fila.style.transition = 'opacity 0.3s';
        fila.style.opacity = '0';
        setTimeout(() => { fila.remove(); filtrar(filtroActivo); }, 300);
    } else {
        alert(d.error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Botones de filtro
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            filtroActivo = btn.getAttribute('data-filtro');
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filtrar();
        });
    });

    // Buscador
    document.getElementById('buscador').addEventListener('input', filtrar);

    // Contador inicial
    filtrar();
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        document.getElementById('modalOverlay').classList.remove('open');
    }
});