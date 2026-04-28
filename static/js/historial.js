// ── FILTROS ─────────────────────────────────────────────
const filterBtns = document.querySelectorAll(".filter-btn");
const buscador   = document.getElementById("buscador");

function actualizarTabla() {
    const filtroActivo = document.querySelector(".filter-btn.active")?.dataset.filtro || "todos";
    const texto        = buscador.value.toLowerCase();
    const filas        = document.querySelectorAll(".fila-archivo");

    let visibles = 0;
    filas.forEach(fila => {
        const tipo   = fila.dataset.tipo;
        const nombre = fila.dataset.nombre.toLowerCase();

        const pasaFiltro  = filtroActivo === "todos" || tipo === filtroActivo;
        const pasaBusqueda = nombre.includes(texto);

        if (pasaFiltro && pasaBusqueda) {
            fila.style.display = "";
            visibles++;
        } else {
            fila.style.display = "none";
        }
    });

    // Contador
    const contador = document.getElementById("contadorVisible");
    if (contador) contador.textContent = `(${visibles})`;

    // Fila vacía
    const filaVacia = document.getElementById("filaVacia");
    if (filaVacia) filaVacia.style.display = visibles === 0 ? "" : "none";
}

filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        filterBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        actualizarTabla();
    });
});

buscador.addEventListener("input", actualizarTabla);

// Contador inicial
actualizarTabla();

// ── ELIMINAR ────────────────────────────────────────────
function eliminar(nombre, btnEl) {
    if (!confirm(`¿Eliminar "${nombre}"?`)) return;

    btnEl.disabled = true;
    btnEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    fetch(`/historial/eliminar/${encodeURIComponent(nombre)}`, { method: "POST" })
        .then(r => r.json())
        .then(data => {
            if (!data.ok) {
                alert("Error al eliminar: " + (data.error || ""));
                btnEl.disabled = false;
                btnEl.innerHTML = '<i class="fas fa-trash"></i> Eliminar';
                return;
            }

            // Solo elimina la fila exacta del archivo presionado
            const fila = btnEl.closest(".fila-archivo");
            if (fila) fila.remove();

            actualizarTabla();
        })
        .catch(() => {
            alert("Error de conexión");
            btnEl.disabled = false;
            btnEl.innerHTML = '<i class="fas fa-trash"></i> Eliminar';
        });
}

// ── MODAL IMAGEN ─────────────────────────────────────────
document.querySelectorAll(".fila-archivo[data-tipo='png']").forEach(fila => {
    fila.querySelector(".td-nombre")?.addEventListener("click", () => {
        const nombre = fila.dataset.nombre;
        document.getElementById("modalNombre").textContent = nombre;
        document.getElementById("modalImg").src = `/static/emg/${nombre}`;
        document.getElementById("modalOverlay").style.display = "flex";
    });
});

function cerrarModal(e) {
    if (!e || e.target === document.getElementById("modalOverlay") || e.currentTarget?.classList.contains("modal-close")) {
        document.getElementById("modalOverlay").style.display = "none";
        document.getElementById("modalImg").src = "";
    }
}