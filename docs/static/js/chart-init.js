/**
 * Chart initialization for DAB project website
 * Creates and manages charts for all experiment types
 */

// Chart instances
const charts = {};

// Create a chart
function createChart(chartType) {
    const config = chartData[chartType];
    const canvasId = chartType + 'Chart';
    const canvas = document.getElementById(canvasId);
    
    // Skip if canvas doesn't exist
    if (!canvas) return null;
    
    // Clear previous chart if exists
    if (charts[chartType]) {
        charts[chartType].destroy();
    }
    
    // Create new chart
    charts[chartType] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: config.labels,
            datasets: [{
                label: config.datasetLabel,
                data: config.data,
                backgroundColor: config.backgroundColor,
                borderColor: config.borderColor,
                borderWidth: config.borderWidth || 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500 // Shorter animation to prevent visual glitches
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${config.datasetLabel}: ${context.parsed.y}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    min: config.minScale || 0,
                    max: config.maxScale,
                    ticks: {
                        stepSize: config.stepSize,
                        callback: function(value) {
                            return value + (config.tickSuffix || '');
                        }
                    },
                    title: {
                        display: config.yAxisTitle ? true : false,
                        text: config.yAxisTitle || ''
                    }
                },
                x: {
                    title: {
                        display: config.xAxisTitle ? true : false,
                        text: config.xAxisTitle || ''
                    }
                }
            }
        }
    });
    
    return charts[chartType];
}

// Switch between charts
function switchChart(chartType) {
    console.log("Switching to chart type:", chartType);
    
    // Hide all chart displays
    document.querySelectorAll('.chart-display').forEach(display => {
        display.style.display = 'none';
    });
    
    // Deactivate all tabs
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected chart display
    const display = document.getElementById(`${chartType}-display`);
    if (display) {
        console.log("Found display element:", display.id);
        display.style.display = 'block';
    } else {
        console.warn(`Display element for ${chartType} not found`);
    }
    
    // Activate selected tab
    const tab = document.querySelector(`.chart-tab[data-chart="${chartType}"]`);
    if (tab) {
        tab.classList.add('active');
    }
    
    // Create or update charts for the selected experiment type
    setTimeout(() => {
        console.log("Creating charts for:", chartType);
        if (chartType === 'sentiment') {
            // Control metrics
            createChart('sentimentInternal');
            createChart('sentimentExternal');
            // Fluency metrics
            createChart('sentimentPerplexity');
            createChart('sentimentCola');
        } 
        else if (chartType === 'toxicity') {
            // Control metrics
            createChart('toxicityInternal');
            createChart('toxicityExternal');
            // Fluency metrics
            createChart('toxicityPerplexity');
            createChart('toxicityCola');
        }
        else if (chartType === 'keyword') {
            // Control metrics
            createChart('keywordInclusion');
            createChart('keywordBertscore');
            // Fluency metrics
            createChart('keywordPerplexity');
            createChart('keywordCola');
        }
    }, 100);
}

// Initialize all charts and tab functionality
function initializeCharts() {
    console.log("Initializing charts and tab functionality");
    
    // Make sure all chart displays are in the right state initially
    document.querySelectorAll('.chart-display').forEach(display => {
        if (display.id === 'sentiment-display') {
            display.style.display = 'block';
        } else {
            display.style.display = 'none';
        }
    });
    
    // Set up tab click handlers
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const chartType = this.getAttribute('data-chart');
            console.log("Tab clicked:", chartType);
            switchChart(chartType);
        });
    });
    
    // Initialize with sentiment chart set (or get active tab if it exists)
    setTimeout(() => {
        console.log("Creating initial charts");
        const activeTab = document.querySelector('.chart-tab.active');
        if (activeTab) {
            const chartType = activeTab.getAttribute('data-chart');
            console.log("Active tab found:", chartType);
            switchChart(chartType);
        } else {
            // Default to sentiment
            console.log("No active tab found, defaulting to sentiment");
            const sentimentTab = document.querySelector('.chart-tab[data-chart="sentiment"]');
            if (sentimentTab) {
                sentimentTab.classList.add('active');
                switchChart('sentiment');
            }
        }
    }, 300);
}

// Resize handler to redraw charts if window size changes
window.addEventListener('resize', function() {
    const activeChartType = document.querySelector('.chart-tab.active')?.getAttribute('data-chart');
    if (activeChartType) {
        switchChart(activeChartType);
    }
});