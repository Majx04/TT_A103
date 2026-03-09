let pausado=false;

const ctx=document.getElementById("grafica").getContext("2d");

const chart=new Chart(ctx,{
type:"line",
data:{
labels:[],
datasets:[{
label:"",
data:[],
borderWidth:1,
pointRadius:0
}]
},
options:{
animation:false,
responsive:true,
maintainAspectRatio:false,
scales:{
y:{
title:{
display:true,
text:"Voltaje (V)"
}
},
x:{
title:{
display:true,
text:"Muestras"
}
}
}
}
});

function actualizar(){

fetch("/datos_emg")
.then(r=>r.json())
.then(data=>{

chart.data.labels=data.map((_,i)=>i);
chart.data.datasets[0].data=data;

/* ajuste dinámico del eje Y */

let min=Math.min(...data);
let max=Math.max(...data);

chart.options.scales.y.min=min-0.1;
chart.options.scales.y.max=max+0.1;

chart.update();

});

}

setInterval(actualizar,50);


/* pausa */

function toggleEMG(){

fetch("/toggle_emg",{method:"POST"})
.then(r=>r.json())
.then(data=>{

pausado=data.pausado;

let estado=document.getElementById("estado");
let btn=document.getElementById("btnPausa");
let guardar=document.getElementById("btnGuardar");

if(pausado){

estado.innerText="PAUSADO";
estado.className="estado-pausado";

btn.innerText="Reanudar";

guardar.disabled=false;

}else{

estado.innerText="Adquisición activa";
estado.className="estado-activo";

btn.innerText="Pausar";

guardar.disabled=true;

}

});

}


/* guardar */

function guardarEMG(){

let nombre=document.getElementById("nombre").value;

fetch("/guardar_emg",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({nombre:nombre})

})
.then(r=>r.json())
.then(data=>{

if(data.error){

alert(data.error);

}else{

alert("Archivo guardado como: "+data.ok);

}

});

}


/* dummy */

function crearCarpeta(){
alert("La carpeta se define en el servidor Flask");
}

function seleccionarCarpeta(){
alert("La carpeta se define en el servidor Flask");
}