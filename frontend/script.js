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

    // Initialize Bootstrap tooltips only if Bootstrap JS is available.
    if (window.bootstrap?.Tooltip) {
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
            .forEach(element => {
            new window.bootstrap.Tooltip(element);
            });
    }
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
    const emptyPreview = document.getElementById('resultEmpty');
    const resultDiv = document.getElementById('result');
    const singleResult = document.getElementById('singleResult');
    const bulkResult = document.getElementById('bulkResult');
    const numbersDiv = document.getElementById('numbers');
    const bulkNumbersDiv = document.getElementById('bulkNumbers');
    const messageDiv = document.getElementById('message');
    const errorDiv = document.getElementById('error');
    const resultInfo = document.getElementById('resultInfo');

    const button = document.querySelector(
        'button[onclick="getSuggestion()"]'
    );

    if (!resultDiv || !numbersDiv || !bulkNumbersDiv) {
        console.error('Suggestion UI is missing required result elements.');
        showToast('errorToast', 'The results area could not be found.');
        return;
    }

    if (button?.disabled) return;

    const originalButtonContent = button?.innerHTML || 'Generate Suggestion';

    const variant = document.getElementById('variant')?.value;
    const modelType = document.getElementById('modelType')?.value ?? '';
    const mode = document.getElementById('mode')?.value;
    const bulkCount = parseInt(
        document.getElementById('bulkCount')?.value || '1',
        10
    );

    if (!variant || !mode || !Number.isInteger(bulkCount) || bulkCount < 1) {
        if (errorDiv) {
            errorDiv.textContent = 'Please select valid lottery settings.';
            errorDiv.style.display = 'block';
        }
        showToast('errorToast', 'Please check your lottery settings.');
        return;
    }

    const fullVariant = variant + modelType;
    let displayedResults = false;

    // Prepare the interface for a new request.
    if (button) {
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        button.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2"
                  role="status" aria-hidden="true"></span>
            Generating...
        `;
    }

    // Show the placeholder while a new suggestion is being generated.
    if (emptyPreview) {
        emptyPreview.style.display = '';
    }

    resultDiv.style.display = 'none';

    if (singleResult) singleResult.style.display = 'none';
    if (bulkResult) bulkResult.style.display = 'none';
    if (resultInfo) resultInfo.style.display = 'none';
    if (messageDiv) messageDiv.style.display = 'none';
    if (errorDiv) errorDiv.style.display = 'none';

    numbersDiv.replaceChildren();
    bulkNumbersDiv.replaceChildren();

    // Check and parse the backend response.
    async function requestSuggestion() {
        const response = await fetch(
            `/suggest/${encodeURIComponent(fullVariant)}/${encodeURIComponent(mode)}`,
            { method: 'POST' }
        );

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                `The server returned an invalid response (HTTP ${response.status}).`
            );
        }

        if (!response.ok && !data?.suggested && !data?.error) {
            throw new Error(
                `Request failed with HTTP ${response.status} ${response.statusText}.`
            );
        }

        if (!data || typeof data !== 'object') {
            throw new Error('The server returned an empty or invalid response.');
        }

        return data;
    }

    function displaySingleSuggestion(data) {
        if (!Array.isArray(data.suggested) || data.suggested.length === 0) {
            throw new Error('The server did not return any suggested numbers.');
        }

        numbersDiv.replaceChildren(
            ...data.suggested.map(number => createNumberBall(number))
        );

        resultDiv.style.display = 'block';

        if (singleResult) singleResult.style.display = 'block';
        if (bulkResult) bulkResult.style.display = 'none';

        // Probability data may be absent; the numbers should still display.
        if (Array.isArray(data.probabilities) && data.probabilities.length) {
            try {
                updateChart(data);
            } catch (chartError) {
                console.error('Could not update the probability chart:', chartError);
            }
        }

        try {
            updateResultExplanation(
                data, data.suggested, variant, modelType, mode, false
            );
        } catch (explanationError) {
            console.error('Could not update result details:', explanationError);
        }

        displayedResults = true;
    }

    try {
        if (bulkCount === 1) {
            // Single suggestion.
            const data = await requestSuggestion();

            if (data.status === 'retry') {
                if (messageDiv) {
                    messageDiv.textContent =
                        data.message || 'The model requested a retry.';
                    messageDiv.style.display = 'block';
                }

                if (Array.isArray(data.suggested) && data.suggested.length) {
                    displaySingleSuggestion(data);
                    if (emptyPreview) emptyPreview.style.display = 'none';

                    incrementStreak();
                    addToHistory(data.suggested, variant, modelType, mode);
                    updateResultTitle();
                } else {
                    showToast('errorToast', data.message || 'Please try again.');
                }

                return;
            }

            if (data.error && !Array.isArray(data.suggested)) {
                throw new Error(data.error);
            }

            if (data.error && !data.suggested?.length) {
                throw new Error(data.error);
            }

            displaySingleSuggestion(data);

            if (emptyPreview) emptyPreview.style.display = 'none';

            incrementStreak();
            addToHistory(data.suggested, variant, modelType, mode);
            updateResultTitle();

            showToast(
                data.error ? 'errorToast' : 'successToast',
                data.error
                    ? 'Numbers were returned, but the server reported a warning.'
                    : 'Your suggestion has been generated.'
            );

        } else {
            // Bulk suggestions.
            const suggestions = [];
            let lastProbabilities = [];

            for (let i = 0; i < bulkCount; i++) {
                const data = await requestSuggestion();

                if (data.error && !data.suggested?.length) {
                    throw new Error(
                        `Suggestion ${i + 1}: ${data.error}`
                    );
                }

                if (!Array.isArray(data.suggested) || !data.suggested.length) {
                    throw new Error(
                        `Suggestion ${i + 1} did not contain any numbers.`
                    );
                }

                suggestions.push(data.suggested);

                if (
                    Array.isArray(data.probabilities) &&
                    data.probabilities.length
                ) {
                    lastProbabilities = data.probabilities;
                }
            }

            // Build the bulk result cards.
            suggestions.forEach((nums, index) => {
                const col = document.createElement('div');
                col.className = 'col-md-6 col-lg-4';

                const card = document.createElement('div');
                card.className = 'card p-3 mb-3';

                const heading = document.createElement('h6');
                heading.className = 'text-center';
                heading.textContent = `Suggestion ${index + 1}`;

                const ballContainer = document.createElement('div');
                ballContainer.className =
                    'd-flex flex-wrap justify-content-center gap-1';

                nums.forEach(number => {
                    ballContainer.appendChild(createNumberBall(number));
                });

                card.append(heading, ballContainer);
                col.appendChild(card);
                bulkNumbersDiv.appendChild(col);
            });

            resultDiv.style.display = 'block';

            if (bulkResult) bulkResult.style.display = 'block';
            if (singleResult) singleResult.style.display = 'none';

            if (emptyPreview) emptyPreview.style.display = 'none';

            if (lastProbabilities.length) {
                try {
                    updateChart({
                        probabilities: lastProbabilities,
                        suggested: suggestions[0]
                    });
                } catch (chartError) {
                    console.error('Could not update the probability chart:', chartError);
                }
            }

            try {
                updateResultExplanation(
                    { probabilities: lastProbabilities },
                    suggestions[0],
                    variant,
                    modelType,
                    mode,
                    true,
                    suggestions.length
                );
            } catch (explanationError) {
                console.error('Could not update result details:', explanationError);
            }

            incrementStreak();
            addBulkToHistory(suggestions, variant, modelType, mode);
            updateResultTitle();

            displayedResults = true;
            showToast('successToast', `${suggestions.length} suggestions generated.`);
        }

    } catch (err) {
        console.error('Suggestion generation failed:', err);

        if (!displayedResults) {
            resultDiv.style.display = 'none';

            if (emptyPreview) emptyPreview.style.display = '';

            if (errorDiv) {
                errorDiv.textContent =
                    err instanceof TypeError
                        ? 'Unable to connect to the API. Check your backend server and network connection.'
                        : (err.message || 'An unexpected error occurred.');

                errorDiv.style.display = 'block';
            }
        }

        showToast(
            'errorToast',
            err.message || 'Failed to generate suggestions.'
        );

    } finally {
        if (button) {
            button.disabled = false;
            button.removeAttribute('aria-busy');
            button.innerHTML = originalButtonContent;
        }
    }
}




function showToast(type, message) {
    // Accept both existing calls:
    // showToast('successToast')
    // showToast('errorToast')
    // and semantic calls:
    // showToast('success', 'Generated successfully!')
    // showToast('error', 'Something went wrong.')

    const isSuccess =
        type === 'success' || type === 'successToast';

    const toastId = isSuccess ? 'successToast' : 'errorToast';
    const toast = document.getElementById(toastId);

    if (!toast) {
        console[isSuccess ? 'log' : 'error'](
            message || (isSuccess ? 'Success' : 'Error')
        );
        return;
    }

    if (message) {
        // Preserve any icons or markup inside the toast.
        const messageElement = toast.querySelector(
            '.toast-body, [data-toast-message]'
        );

        if (messageElement) {
            messageElement.textContent = message;
        } else {
            toast.textContent = message;
        }
    }

    document.querySelectorAll('.toast').forEach(item => {
        item.classList.remove('show');
    });

    toast.classList.add('show');

    if (toast._dismissTimer) {
        clearTimeout(toast._dismissTimer);
    }

    toast._dismissTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3500);
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