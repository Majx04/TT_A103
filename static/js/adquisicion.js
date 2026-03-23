let pausado = false;
const ctx = document.getElementById("grafica").getContext("2d");

// Inicialización de la gráfica
const chart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Señal EMG",
            data: [],
            borderColor: "#5b8fb9",
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false
        }]
    },
    options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { title: { display: true, text: "Voltaje (V)" } },
            x: { title: { display: true, text: "Muestras" }, ticks: { display: false } }
        }
    }
});

// Función para obtener datos
function actualizar() {
    if (pausado) return; // No hace fetch si está pausado

    fetch("/datos_emg")
        .then(r => r.json())
        .then(data => {
            if (data.length > 0) {
                chart.data.labels = data.map((_, i) => i);
                chart.data.datasets[0].data = data;

                // Auto-escala simple
                const min = Math.min(...data);
                const max = Math.max(...data);
                chart.options.scales.y.min = min - 0.1;
                chart.options.scales.y.max = max + 0.1;

                chart.update('none'); // Update sin animaciones para fluidez
            }
        })
        .catch(err => console.error("Error obteniendo datos:", err));
}

// Ejecutar cada 50ms
setInterval(actualizar, 50);

// Función Pausar / Reanudar
function toggleEMG() {
    fetch("/toggle_emg", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            pausado = data.pausado;
            const estado = document.getElementById("estado");
            const btn = document.getElementById("btnPausa");
            const btnGuardar = document.getElementById("btnGuardar");

            if (pausado) {
                estado.innerText = "● Adquisición Detenida";
                estado.className = "estado pausado";
                btn.innerText = "Reanudar";
                btnGuardar.disabled = false;
            } else {
                estado.innerText = "● Adquisición activa";
                estado.className = "estado activo";
                btn.innerText = "Pausar";
                btnGuardar.disabled = true;
            }
        });
}

// Función Guardar
function guardarEMG() {
    const nombreInput = document.getElementById("nombre");
    const nombre = nombreInput.value.trim();

    if (!nombre) {
        alert("Por favor, ingresa un nombre para la muestra.");
        return;
    }

    fetch("/guardar_emg", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nombre: nombre })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            alert("Éxito: Archivo guardado como " + data.ok);
            nombreInput.value = ""; // Limpiar input
        }
    })
    .catch(err => alert("Error al conectar con el servidor"));
}