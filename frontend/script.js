// Lotto AI-X Gen Script

let chart = null;
let streak = 0;

window.addEventListener("DOMContentLoaded", () => {
    createPlaceholderBalls();
    initializeChart();
    loadTheme();
    loadStreak();
    loadFavorites();
    updateSelectionSummary();
    updateHistoryState();
    updateFavoritesState();

    ['variant', 'modelType', 'mode', 'bulkCount'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', updateSelectionSummary);
        }
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

function createPlaceholderBalls() {
    const numbersDiv = document.getElementById('numbers');
    numbersDiv.innerHTML = '';
    for (let i = 0; i < 6; i++) {
        const ball = document.createElement('div');
        ball.className = 'number-ball placeholder';
        ball.textContent = '?';
        numbersDiv.appendChild(ball);
    }
}

function initializeChart() {
    const ctx = document.getElementById('probChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["?", "?", "?", "?", "?", "?"],
            datasets: [{
                label: 'Waiting for prediction...',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(200,200,200,0.6)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100 }
            }
        }
    });
}

function createNumberBall(number) {
    const ball = document.createElement('div');
    ball.className = 'number-ball';
    ball.textContent = number;
    ball.onclick = () => showBallTooltip(ball, number);
    return ball;
}

function showBallTooltip(ball, number) {
    // Simple tooltip for probability (placeholder)
    alert(`Number ${number}: Probability details coming soon!`);
}

function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeButton(newTheme);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeButton(savedTheme);
}

function updateThemeButton(theme) {
    const button = document.getElementById('themeToggle');
    if (theme === 'dark') {
        button.innerHTML = '<i class="bi bi-sun-fill"></i> Light Mode';
    } else {
        button.innerHTML = '<i class="bi bi-moon-fill"></i> Dark Mode';
    }
}

function formatLabel(text) {
    return text
        .replace(/_/g, '/')
        .replace(/-([a-z])/g, (_, char) => ' ' + char.toUpperCase())
        .replace(/\b\w/g, char => char.toUpperCase());
}

function updateSelectionSummary() {
    const variant = document.getElementById('variant').value;
    const modelType = document.getElementById('modelType').value || 'ODE';
    const mode = document.getElementById('mode').value;
    const bulkCount = parseInt(document.getElementById('bulkCount').value, 10);
    const modelLabel = modelType === '' ? 'ODE' : 'Hybrid';
    const summaryText = `Selected: ${variant.replace('_', '/')} · ${modelLabel} · ${formatLabel(mode)} · ${bulkCount} suggestion${bulkCount === 1 ? '' : 's'}`;
    document.getElementById('selectionSummary').textContent = summaryText;

}

function updateHistoryState() {
    const historyList = document.getElementById('historyList');
    const empty = document.getElementById('historyEmpty');
    if (historyList.children.length > 0) {
        historyList.style.display = 'block';
        empty.style.display = 'none';
    } else {
        historyList.style.display = 'none';
        empty.style.display = 'block';
    }
}

function updateFavoritesState() {
    const favoritesList = document.getElementById('favoritesList');
    const empty = document.getElementById('favoritesEmpty');
    if (favoritesList.children.length > 0) {
        favoritesList.style.display = 'block';
        empty.style.display = 'none';
    } else {
        favoritesList.style.display = 'none';
        empty.style.display = 'block';
    }
}

function toggleHistoryVisibility() {
    const historyList = document.getElementById('historyList');
    const empty = document.getElementById('historyEmpty');
    if (historyList.style.display === 'block') {
        historyList.style.display = 'none';
        empty.style.display = 'block';
    } else if (historyList.children.length > 0) {
        historyList.style.display = 'block';
        empty.style.display = 'none';
    }
}

function loadStreak() {
    streak = parseInt(localStorage.getItem('streak')) || 0;
    document.getElementById('streakCount').textContent = streak;
}

function incrementStreak() {
    streak++;
    localStorage.setItem('streak', streak);
    document.getElementById('streakCount').textContent = streak;
}

function updateResultTitle() {
    const variant = document.getElementById('variant').value.replace('_', '/');
    const modelType = document.getElementById('modelType').value;
    const mode = document.getElementById('mode').value;
    const bulkCount = parseInt(document.getElementById('bulkCount').value);

    const modelLabel = modelType === '' ? 'ODE' : 'Hybrid';
    const modeLabel = formatLabel(mode);

    let title = `AI Suggestions — ${variant} · ${modelLabel} · ${modeLabel}`;

    if (bulkCount > 1) {
        title += ` · ${bulkCount} picks`;
    }

    document.getElementById("resultTitle").innerHTML =
        `<i class="bi bi-trophy-fill me-2"></i>${title}`;

    console.log("updateResultTitle fired");
console.log({
  variant: document.getElementById('variant').value,
  modelType: document.getElementById('modelType').value,
  mode: document.getElementById('mode').value,
  bulkCount: document.getElementById('bulkCount').value
});
}

async function getSuggestion() {
    const variant = document.getElementById('variant').value;
    const modelType = document.getElementById('modelType').value;
    const mode = document.getElementById('mode').value;
    const bulkCount = parseInt(document.getElementById('bulkCount').value);
    const fullVariant = variant + modelType;

    const resultDiv = document.getElementById('result');
    const singleResult = document.getElementById('singleResult');
    const bulkResult = document.getElementById('bulkResult');
    const numbersDiv = document.getElementById('numbers');
    const bulkNumbersDiv = document.getElementById('bulkNumbers');
    const messageDiv = document.getElementById('message');
    const errorDiv = document.getElementById('error');
    const resultInfo = document.getElementById('resultInfo');
    const button = document.querySelector('button[onclick="getSuggestion()"]');

    
    // Show loading
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
    resultDiv.style.display = 'none';
    singleResult.style.display = 'none';
    bulkResult.style.display = 'none';
    if (resultInfo) resultInfo.style.display = 'none';
    messageDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    numbersDiv.innerHTML = '';
    bulkNumbersDiv.innerHTML = '';
    

    try {
        if (bulkCount === 1) {
            // Single suggestion
            const response = await fetch(`/suggest/${fullVariant}/${mode}`, { method: 'POST' });
            const data = await response.json();

            if (data.status === "retry") {
                messageDiv.style.display = 'block';
                messageDiv.textContent = data.message;
                if (data.suggested) {
                    // Still display and add to history even on retry
                    resultDiv.style.display = 'block';
                    singleResult.style.display = 'block';
                    data.suggested.forEach(num => numbersDiv.appendChild(createNumberBall(num)));
                    updateChart(data);
                    incrementStreak();
                    addToHistory(data.suggested, variant, modelType, mode);
                }
                return;
            }
            if (data.error) {
                if (!data.suggested) {
                    errorDiv.style.display = 'block';
                    errorDiv.textContent = data.error;
                    showToast('errorToast');
                    return;
                } else {
                    // Display anyway if suggested provided despite error
                    resultDiv.style.display = 'block';
                    singleResult.style.display = 'block';
                    data.suggested.forEach(num => numbersDiv.appendChild(createNumberBall(num)));

                    updateChart(data);
                    incrementStreak();
                    addToHistory(data.suggested, variant, modelType, mode);
                    return;
                }
            }

            // Success case
            resultDiv.style.display = 'block';
            singleResult.style.display = 'block';
            data.suggested.forEach(num => numbersDiv.appendChild(createNumberBall(num)));
            updateChart(data);
            updateResultExplanation(data, data.suggested, variant, modelType, mode, false);
            
            incrementStreak();
            showToast('successToast');
            addToHistory(data.suggested, variant, modelType, mode);
        } else {
            // Bulk suggestions
            const suggestions = [];
            let lastProbabilities = [];
            for (let i = 0; i < bulkCount; i++) {
                const response = await fetch(`/suggest/${fullVariant}/${mode}`, { method: 'POST' });
                const data = await response.json();
                if (data.error && !data.suggested) {
                    errorDiv.style.display = 'block';
                    errorDiv.textContent = `Error on suggestion ${i+1}: ${data.error}`;
                    showToast('errorToast');
                    return;
                }
                suggestions.push(data.suggested || []);
                lastProbabilities = data.probabilities || lastProbabilities;
            }

            // Display bulk results
            resultDiv.style.display = 'block';
            bulkResult.style.display = 'block';
            
            suggestions.forEach((nums, index) => {
                const col = document.createElement('div');
                col.className = 'col-md-6 col-lg-4';
                col.innerHTML = `
                    <div class="card p-3 mb-3">
                        <h6 class="text-center">Suggestion ${index + 1}</h6>
                        <div class="d-flex flex-wrap justify-content-center gap-1">
                            ${nums.map(num => `<div class="number-ball">${num}</div>`).join('')}
                        </div>
                    </div>
                `;
                bulkNumbersDiv.appendChild(col);
            });
            if (lastProbabilities.length) {
                updateChart({ probabilities: lastProbabilities, suggested: suggestions[0] || [] });
            }
            updateResultExplanation({ probabilities: lastProbabilities }, suggestions[0] || [], variant, modelType, mode, true, suggestions.length);

            // Increment streak for bulk
            incrementStreak();

            // Show success toast
            showToast('successToast');

            // Add bulk to history (as a group)
            addBulkToHistory(suggestions, variant, modelType, mode);
        }

        updateResultTitle();
    } catch (err) {
        console.error('Fetch error:', err);
        errorDiv.style.display = 'block';
        if (err.name === 'TypeError' && err.message.includes('fetch')) {
            errorDiv.textContent = 'Network error: Unable to connect to the API. Is the backend server running?';
        } else {
            errorDiv.textContent = `Failed to fetch suggestion: ${err.message}`;
        }
        showToast('errorToast');
    } finally {
        // Reset button
        button.disabled = false;
        button.innerHTML = 'Get Suggestion';
        
    }

    
}

function showToast(toastId) {
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
}

function updateChart(data) {
    const ctx = document.getElementById('probChart').getContext('2d');
    const labels = data.probabilities.map((_, i) => i + 1);
    const probabilities = data.probabilities.map(p => (p * 100).toFixed(2));

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: probabilities,
                backgroundColor: labels.map(n => data.suggested && data.suggested.includes(n) ? '#e74c3c' : '#3498db')
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: Math.max(...probabilities) * 1.2 }
            }
        }
    });

    // Note: addToHistory is called in getSuggestion, not here
}

function updateResultExplanation(data, suggested, variant, modelType, mode, isBulk = false, bulkCount = 1) {
    const info = document.getElementById('resultInfo');
    const summaryText = document.getElementById('resultSummaryText');
    const confidence = document.getElementById('resultConfidence');
    const tip = document.getElementById('resultTip');

    const variantLabel = variant.replace('_', '/');
    const modelLabel = modelType === '' ? 'ODE' : 'Hybrid';
    const modeLabel = formatLabel(mode);
    const numbersText = suggested.length ? suggested.join(', ') : 'No suggestion available';
    const topProb = data.probabilities && data.probabilities.length ? Math.max(...data.probabilities) * 100 : 0;

    if (isBulk) {
        summaryText.textContent = `Generated ${bulkCount} suggestions for ${variantLabel} using ${modelLabel} in ${modeLabel} mode. First suggestion: ${numbersText}.`;
    } else {
        summaryText.textContent = `Suggested numbers for ${variantLabel} using ${modelLabel} in ${modeLabel} mode: ${numbersText}.`;
    }

    confidence.textContent = topProb ? `Top chance ${topProb.toFixed(1)}%` : 'Probability data unavailable';
    tip.textContent = 'Interpret the chart as relative strength: taller bars indicate higher estimated likelihood, with suggested numbers highlighted. Use this for guidance only.';
    info.style.display = 'block';
}

function addBulkToHistory(suggestions, variant, model, mode) {
    const historyList = document.getElementById('historyList');
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex justify-content-between align-items-center';
    const bulkText = suggestions.map(nums => nums.join(',')).join(' | ');
    li.innerHTML = `
        <span><strong>Bulk (${suggestions.length}):</strong> ${bulkText} - ${variant} ${model} (${mode}) - ${new Date().toLocaleString()}</span>
        <button class="btn btn-outline-warning btn-sm" onclick="addBulkToFavorites('${JSON.stringify(suggestions)}', '${variant}', '${model}', '${mode}')">⭐ Favorite</button>
    `;
    historyList.appendChild(li);
    updateHistoryState();

    // Limit to last 5
    while (historyList.children.length > 5) {
        historyList.removeChild(historyList.firstChild);
    }
}

function addToHistory(suggested, variant, model, mode) {
    const historyList = document.getElementById('historyList');
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex justify-content-between align-items-center';
    const numbersStr = suggested.join(', ');
    li.innerHTML = `
        <span><strong>Single:</strong> ${numbersStr} - ${variant} ${model} (${mode}) - ${new Date().toLocaleString()}</span>
        <button class="btn btn-outline-warning btn-sm" onclick="addToFavorites('${numbersStr}', '${variant}', '${model}', '${mode}')">⭐ Favorite</button>
    `;
    historyList.appendChild(li);
    updateHistoryState();

    // Limit to last 5
    while (historyList.children.length > 5) {
        historyList.removeChild(historyList.firstChild);
    }
}

function addBulkToFavorites(suggestionsStr, variant, model, mode) {
    const favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    const suggestions = JSON.parse(suggestionsStr);
    const favorite = { numbers: suggestions, variant, model, mode, timestamp: new Date().toISOString(), isBulk: true };
    favorites.push(favorite);
    localStorage.setItem('favorites', JSON.stringify(favorites));
    loadFavorites();
    showToast('successToast');
}

function addToFavorites(numbersStr, variant, model, mode) {
    const favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    const favorite = { numbers: numbersStr.split(',').map(n => parseInt(n)), variant, model, mode, timestamp: new Date().toISOString() };
    favorites.push(favorite);
    localStorage.setItem('favorites', JSON.stringify(favorites));
    loadFavorites();
    showToast('successToast'); // Reuse for favorite added
}

function loadFavorites() {
    const favoritesList = document.getElementById('favoritesList');
    favoritesList.innerHTML = '';
    const favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    favorites.forEach(fav => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';
        const numbersText = fav.isBulk ? `Bulk (${fav.numbers.length}): ${fav.numbers.map(nums => nums.join(',')).join(' | ')}` : fav.numbers.join(', ');
        li.innerHTML = `
            <span><strong>${numbersText}</strong> - ${fav.variant} ${fav.model} (${fav.mode}) - ${new Date(fav.timestamp).toLocaleString()}</span>
            <button class="btn btn-outline-danger btn-sm" onclick="removeFavorite('${fav.timestamp}')">Remove</button>
        `;
        favoritesList.appendChild(li);
    });
    updateFavoritesState();
}

function removeFavorite(timestamp) {
    const favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    const updated = favorites.filter(fav => fav.timestamp !== timestamp);
    localStorage.setItem('favorites', JSON.stringify(updated));
    loadFavorites();
}

function clearFavorites() {
    localStorage.removeItem('favorites');
    loadFavorites();
}

function clearHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    updateHistoryState();
}

function exportHistory() {
    const historyList = document.getElementById('historyList');
    const items = historyList.querySelectorAll('li');
    if (items.length === 0) {
        alert('No history to export.');
        return;
    }

    let csv = 'Type,Numbers,Variant,Model,Mode,Timestamp\n';
    items.forEach(item => {
        const text = item.querySelector('span').textContent;
        // Parse the text: e.g., "Bulk (5): 1,2,3,4,5,6 | 7,8,9,10,11,12 | ... - 6_42  (balance) - 4/27/2026, 12:00:00 PM"
        const parts = text.split(' - ');
        const numbersPart = parts[0];
        const details = parts[1];
        const timestamp = parts[2];
        const isBulk = numbersPart.includes('Bulk');
        csv += `"${isBulk ? 'Bulk' : 'Single'}","${numbersPart}","${details}","${timestamp}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lotto_history.csv';
    a.click();
    URL.revokeObjectURL(url);
}

function exportFavorites() {
    const favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    if (favorites.length === 0) {
        alert('No favorites to export.');
        return;
    }
    let csv = 'Type,Numbers,Variant,Model,Mode,Timestamp\n';
    favorites.forEach(fav => {
        const numbersText = fav.isBulk ? `Bulk: ${fav.numbers.map(nums => nums.join(',')).join(' | ')}` : fav.numbers.join(',');
        csv += `"${fav.isBulk ? 'Bulk' : 'Single'}","${numbersText}","${fav.variant}","${fav.model}","${fav.mode}","${fav.timestamp}"\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lotto_favorites.csv';
    a.click();
    URL.revokeObjectURL(url);
}

function shareOnTwitter() {
    let numbers = '';
    if (document.getElementById('singleResult').style.display !== 'none') {
        numbers = Array.from(document.querySelectorAll('#numbers .number-ball')).map(ball => ball.textContent).join(', ');
    } else {
        // For bulk, share the first suggestion
        const firstCard = document.querySelector('#bulkNumbers .card');
        if (firstCard) {
            numbers = Array.from(firstCard.querySelectorAll('.number-ball')).map(ball => ball.textContent).join(', ');
        }
    }
    const text = `My AI lotto pick: ${numbers} #LottoAI #LuckyNumbers`;
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`;
    window.open(url, '_blank');
}

function shareOnFacebook() {
    let numbers = '';
    if (document.getElementById('singleResult').style.display !== 'none') {
        numbers = Array.from(document.querySelectorAll('#numbers .number-ball')).map(ball => ball.textContent).join(', ');
    } else {
        // For bulk, share the first suggestion
        const firstCard = document.querySelector('#bulkNumbers .card');
        if (firstCard) {
            numbers = Array.from(firstCard.querySelectorAll('.number-ball')).map(ball => ball.textContent).join(', ');
        }
    }
    const text = `My AI lotto pick: ${numbers}`;
    const url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(window.location.href)}&quote=${encodeURIComponent(text)}`;
    window.open(url, '_blank');
}