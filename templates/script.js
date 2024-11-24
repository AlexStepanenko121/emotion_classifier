async function classifyText() {
    const textInput = document.getElementById("text-input").value;
    const resultDiv = document.getElementById('result');
    const spinner = document.getElementById('spinner');

    if (!textInput) {
        resultDiv.innerText = "Please enter some text!";
        return;
    }

    spinner.style.display = 'block';
    resultDiv.innerText = '';

    try {
        const response = await fetch('http://localhost:5000/api/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: textInput }),
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const data = await response.json();

        if (data.emotion_probabilities) {
            const probs = data.emotion_probabilities;


            const filteredEmotions = Object.entries(probs)
                .filter(([, prob]) => prob > 0.01)
                .sort(([, a], [, b]) => b - a);

            let resultText = '';
            if (filteredEmotions.length > 0) {
                resultText += 'Emotion Probabilities (above 1%):\n\n';
                for (const [emotion, prob] of filteredEmotions) {
                    resultText += `<strong>${emotion}:</strong> ${(prob * 100).toFixed(2)}%\n`;
                }
            } else {
                resultText = 'No significant emotions detected.';
            }

            resultDiv.innerHTML = resultText;
        } else {
            resultDiv.innerText = `Error: ${data.error}`;
        }
    } catch (error) {
        console.error("Error:", error);
        resultDiv.innerText = "Error: Something went wrong!";
    } finally {
        spinner.style.display = 'none';
    }
}
