async function predictNews() {
    const newsText = document.getElementById('newsText').value.trim();
    
    if (!newsText) {
        alert('Please enter some text to analyze');
        return;
    }
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: newsText })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Update the UI with the final prediction
        const resultElement = document.querySelector('.result');
        resultElement.textContent = data.final_prediction;
        resultElement.className = 'result ' + (data.final_prediction === 'Fake News' ? 'fake' : 'real');
        
        // Update confidence score
        const confidenceElement = document.querySelector('.confidence');
        confidenceElement.innerHTML = `
            <div class="confidence-score">
                <span class="label">Confidence:</span>
                <span class="value">${(data.confidence[data.final_prediction === 'Fake News' ? 'fake' : 'real'] * 100).toFixed(2)}%</span>
            </div>
        `;
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request. Please try again.');
    }
}

function updatePredictions(predictions) {
    const modelNames = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'gradient_boosting': 'Gradient Boosting',
        'random_forest': 'Random Forest'
    };
    
    for (const [model, prediction] of Object.entries(predictions)) {
        const resultElement = document.querySelector(`#predictions .prediction-card:nth-child(${
            Object.keys(predictions).indexOf(model) + 1
        }) .result`);
        
        if (resultElement) {
            resultElement.textContent = prediction;
            resultElement.className = 'result ' + (prediction === 'Fake News' ? 'fake' : 'real');
        }
    }
} 