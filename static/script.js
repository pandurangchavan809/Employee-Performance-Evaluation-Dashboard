document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("evaluateForm");
    const resultDiv = document.getElementById("predictionResult");

    if(form) {
        form.addEventListener("submit", function(e) {
            const values = Array.from(form.querySelectorAll("input[type='number']")).map(input => parseFloat(input.value));
            const invalid = values.some(v => v < 1 || v > 10);

            if(invalid) {
                e.preventDefault();
                resultDiv.innerHTML = "<p style='color:red;'>All numeric fields must be between 1 and 10!</p>";
            }
        });
    }
});
