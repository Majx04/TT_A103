function entrenar(){

fetch("/entrenar",{
method:"POST"
})
.then(response => response.json())
.then(data => {

document.getElementById("accuracy").innerText = data.accuracy.toFixed(3)
document.getElementById("precision").innerText = data.precision.toFixed(3)
document.getElementById("recall").innerText = data.recall.toFixed(3)
document.getElementById("f1").innerText = data.f1.toFixed(3)

let html = "<table border='1' style='border-collapse:collapse;'>"

data.confusion_matrix.forEach(row => {

html += "<tr>"

row.forEach(val => {
html += "<td style='padding:6px'>" + val + "</td>"
})

html += "</tr>"

})

html += "</table>"

document.getElementById("matriz").innerHTML = html

})
.catch(error=>{
console.log("Error:",error)
})

}