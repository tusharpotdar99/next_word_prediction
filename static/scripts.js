const inputText = document.getElementById('inputText');
const wordSuggestionsDiv = document.getElementById('wordSuggestions');
const sentenceSuggestionsDiv = document.getElementById('sentenceSuggestions');
const displayButton = document.getElementById('displayButton');
const displayArea = document.getElementById('displayArea');

// Fetch suggestions dynamically as the user types
inputText.addEventListener('input', async () => {
    const text = inputText.value;

    if (text.length > 0) {
        const response = await fetch('/api/suggestions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_text: text })
        });

        const data = await response.json();
        showSuggestions(text, data.words, data.sentences);
    } else {
        clearSuggestions(); // Clear suggestions if input is empty
    }
});

// Display suggestions with the typed text included
function showSuggestions(typedText, words, sentences) {
    wordSuggestionsDiv.innerHTML = words
        .map(word => `<div class="suggestion-item">${typedText} ${word}</div>`)
        .join('');

    sentenceSuggestionsDiv.innerHTML = sentences
        .map(sentence => `<div class="suggestion-item">${sentence}</div>`)
        .join('');

    addSuggestionClickEvents();
}

// Add click event to suggestions (for words and sentences)
function addSuggestionClickEvents() {
    document.querySelectorAll('.suggestion-item').forEach(div => {
        div.addEventListener('click', () => {
            inputText.value = div.textContent;
            clearSuggestions(); // Clear suggestions after selection
        });
    });
}

// Clear all suggestions
function clearSuggestions() {
    wordSuggestionsDiv.innerHTML = '';
    sentenceSuggestionsDiv.innerHTML = '';
}

// Display the typed sentence when the button is clicked
displayButton.addEventListener('click', () => {
    const sentence = inputText.value;
    displayArea.textContent = ` ${sentence}`;
    inputText.value = ''; // Clear input field
});
