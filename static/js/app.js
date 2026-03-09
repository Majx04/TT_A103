let chart=null

function entrenar(){

fetch("/entrenar",{
method:"POST"
})
.then(r=>r.json())
.then(data=>{

document.getElementById("accuracy").innerText=data.accuracy.toFixed(3)
document.getElementById("precision").innerText=data.precision.toFixed(3)
document.getElementById("recall").innerText=data.recall.toFixed(3)
document.getElementById("f1").innerText=data.f1.toFixed(3)

dibujarCM(data.confusion_matrix)

})

}


function dibujarCM(cm){

let valores=[
cm[0][0],cm[0][1],cm[0][2],
cm[1][0],cm[1][1],cm[1][2],
cm[2][0],cm[2][1],cm[2][2]
]

const ctx=document.getElementById("cmChart").getContext("2d")

if(chart){
chart.destroy()
}

chart=new Chart(ctx,{
type:"bar",
data:{
labels:[
"0-0","0-1","0-2",
"1-0","1-1","1-2",
"2-0","2-1","2-2"
],
datasets:[{
label:"Confusion Matrix",
data:valores
}]
},
options:{
responsive:true
}
})

}


