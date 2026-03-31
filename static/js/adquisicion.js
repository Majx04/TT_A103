let pausado = false;
const ctx = document.getElementById("grafica").getContext("2d");

// 1. Inicialización de la gráfica
const chart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Señal EMG (Voltaje)",
            data: [],
            borderColor: "#5b8fb9",
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.1
        }]
    },
    options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                min: 0,
                max: 3.5, // Ajustado al VREF de 3.3V de tu Python
                title: { display: true, text: "Voltaje (V)" }
            },
            x: { ticks: { display: false } }
        }
    }
});

// 2. Función para obtener datos (Ruta corregida a /emg/data)
function actualizar() {
    if (pausado) return;

    fetch("/emg/data") // <--- CAMBIO AQUÍ
        .then(r => r.json())
        .then(data => {
            if (Array.isArray(data) && data.length > 0) {
                chart.data.labels = data.map((_, i) => i);
                chart.data.datasets[0].data = data;
                chart.update('none');
            }
        })
        .catch(err => console.error("Error obteniendo datos:", err));
}

setInterval(actualizar, 50);

// 3. Función Pausar / Reanudar (Ruta corregida a /emg/toggle)
function toggleEMG() {
    fetch("/emg/toggle", { method: "POST" }) // <--- CAMBIO AQUÍ
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

// 4. Función Guardar (Ruta corregida a /emg/guardar)
function guardarEMG() {
    const nombreInput = document.getElementById("nombre");
    const nombre = nombreInput.value.trim();

    if (!nombre) {
        alert("Por favor, ingresa un nombre.");
        return;
    }

    // Tu Python usa request.form.get("nombre"), así que enviamos FormData
    const formData = new FormData();
    formData.append("nombre", nombre);

    fetch("/emg/guardar", { // <--- CAMBIO AQUÍ
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            alert("Guardado como: " + data.ok);
            nombreInput.value = "";
        } else {
            alert("Error: " + data.error);
        }
    })
    .catch(err => alert("Error al conectar con el servidor"));
}