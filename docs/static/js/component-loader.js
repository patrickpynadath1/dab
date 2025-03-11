/**
 * Simple component loader for DAB project website
 * Loads HTML fragments and injects them into the page
 */

// Keep track of loaded components
let loadedComponents = 0;
let totalComponents = 0;
let resultsLoaded = false;

document.addEventListener('DOMContentLoaded', function() {
    // Load all components marked with data-component attribute
    const components = document.querySelectorAll('[data-component]');
    totalComponents = components.length;
    
    components.forEach(container => {
        const componentPath = container.getAttribute('data-component');
        
        // Fetch the component HTML
        fetch(componentPath)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to load component: ${componentPath}`);
                }
                return response.text();
            })
            .then(html => {
                // Insert the component HTML
                container.innerHTML = html;
                loadedComponents++;
                
                // Check if this is a results component
                if (componentPath.includes('results')) {
                    resultsLoaded = true;
                }
                
                // Dispatch event to notify component was loaded
                const event = new CustomEvent('component-loaded', {
                    detail: { componentPath, isComplete: loadedComponents === totalComponents }
                });
                document.dispatchEvent(event);
                
                // If all components are loaded, initialize charts
                if (loadedComponents === totalComponents && resultsLoaded) {
                    console.log("All components loaded, initializing charts");
                    setTimeout(() => {
                        if (typeof initializeCharts === 'function') {
                            initializeCharts();
                        }
                    }, 300);
                }
            })
            .catch(error => {
                console.error(error);
                container.innerHTML = `<div class="notification is-danger">
                    Failed to load component: ${componentPath}
                </div>`;
                loadedComponents++;
            });
    });
});