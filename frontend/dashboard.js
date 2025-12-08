const apiBase = window.location.origin;

async function analyzeAudio() {
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please choose an audio file.');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch(`${apiBase}/analyze_audio`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert(`Analysis error: ${err.message}`);
    }
}

function renderResults(data) {
    const results = document.getElementById('results');
    results.innerHTML = `
        <h2>Verdict: ${data.verdict}</h2>
        <p>Risk Score: ${data.final_score.toFixed(2)}</p>
        <p>Classifier: ${data.label} (${data.classifier_score.toFixed(2)})</p>
    `;
    document.getElementById('waveform').src = `data:image/png;base64,${data.visuals.waveform}`;
    document.getElementById('spectrogram').src = `data:image/png;base64,${data.visuals.spectrogram}`;
    document.getElementById('jitter').src = `data:image/png;base64,${data.visuals.jitter}`;
    document.getElementById('artifact').src = `data:image/png;base64,${data.visuals.artifact_map}`;
}

document.getElementById('analyzeBtn').addEventListener('click', analyzeAudio);
