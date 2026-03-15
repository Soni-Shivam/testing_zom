/**
 * Zomato Clone - Frontend Logic
 * Manages routing, global cart state, and API communication with CSAOEngine via FastAPI.
 */

// --- GLOBAL STATE ---
const STATE = {
    userId: 1, 
    cart: [],
    menuData: null,
    currentView: 'home',
    cartTotal: 0,
    currentCuisine: 'North Indian', // Default
    currentRestaurant: 'Spice Heaven' // Default
};

// --- ROUTER ---
async function navigate(viewId, context = {}) {
    console.log(`Navigating to ${viewId}`, context);
    
    if (context.cuisine) STATE.currentCuisine = context.cuisine;
    if (context.restaurant) STATE.currentRestaurant = context.restaurant;

    // Reset menu data if cuisine changed
    if (context.cuisine) STATE.menuData = null;

    // Hide all views
    document.querySelectorAll('.app-view').forEach(v => v.classList.remove('active'));
    
    // Show target view
    const target = document.getElementById(`view-${viewId}`);
    if (target) {
        target.classList.add('active');
        STATE.currentView = viewId;
        
        // Execute view-specific lifecycle hooks
        if (viewId === 'home') {
            await renderHomeView();
        } else if (viewId === 'menu') {
            await renderMenuView();
            updateMenuCartBanner();
        } else if (viewId === 'cart') {
            await renderCartView();
            fetchSmartAddons(); // Trigger Stage 4 Backend Recommendation
        }
        
        // Scroll to top
        target.scrollTop = 0;
    }
}

// --- API FETCHERS ---

// 1. Fetch HTML Templates (Simulating a build process, we load them dynamically here)
async function fetchTemplate(name) {
    const res = await fetch(`/${name}.html`);
    return await res.text();
}

async function fetchMenuData(cuisine) {
    if (!STATE.menuData || (cuisine && cuisine !== STATE.currentCuisine)) {
        const targetCuisine = cuisine || STATE.currentCuisine;
        const res = await fetch(`/menu?cuisine=${encodeURIComponent(targetCuisine)}`);
        STATE.menuData = await res.json();
    }
    return STATE.menuData;
}

async function fetchRestaurants() {
    const res = await fetch('/restaurants');
    return await res.json();
}

async function fetchUserAnalytics() {
    const res = await fetch(`/user/${STATE.userId}/analytics`);
    return await res.json();
}

// --- VIEW CONTROLLERS ---

// 1. HOME VIEW
async function renderHomeView() {
    const container = document.getElementById('view-home');
    
    // Only load HTML once
    if (!container.innerHTML.trim()) {
        const html = await fetchTemplate('home');
        
        // Extract inner content from body to avoid duplicating HTML tags
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        container.innerHTML = doc.body.innerHTML;
    }
    
    // Fetch real User Analytics and Recommendations
    try {
        const data = await fetchUserAnalytics();
        console.log("Analytics loaded:", data.analytics);
        
        // Hydrate the recommendations section dynamically
        const recContainer = container.querySelector('#home-recommendations');
        if (recContainer && data.homepage_recommendations && data.homepage_recommendations.length > 0) {
            recContainer.innerHTML = ''; // clear loading state if any
            
            // Loop over top 3 recommendations
            data.homepage_recommendations.slice(0, 3).forEach((rec) => {
                const imgName = rec.name.replace(/\s/g, '_') + '.png';
                const productImgUrl = `/product-images/${imgName}`;
                const avatarFallback = `https://ui-avatars.com/api/?name=${rec.name.replace(/\s/g, '+')}&background=fceae9&color=E23744&size=200`;
                
                // Add card HTML
                // Make it clickable to go to menu
                recContainer.innerHTML += `
                <div class="bg-surface-container-lowest rounded-xl overflow-hidden custom-shadow cursor-pointer transition-transform active:scale-[0.98]" onclick="navigate('menu')">
                    <div class="relative h-44">
                        <img class="w-full h-full object-cover" src="${productImgUrl}" onerror="this.src='${avatarFallback}'" alt="${rec.name}"/>
                        <div class="absolute top-4 left-0 bg-primary text-on-primary px-3 py-1 rounded-r-lg font-bold text-xs uppercase tracking-wide shadow-sm">
                            Recommended
                        </div>
                    </div>
                    <div class="p-4">
                        <div class="flex justify-between items-start">
                            <div>
                                <h4 class="font-headline text-title-md text-on-surface font-bold leading-tight">${rec.name}</h4>
                                <p class="text-on-surface-variant font-medium text-sm mt-1">₹${rec.price.toFixed(2)}</p>
                            </div>
                            <div class="bg-tertiary/10 text-tertiary px-2 py-1 rounded flex items-center gap-1 shadow-sm">
                                <span class="text-xs font-black">${(Math.min(5.0, 4.0 + (rec.score*10))).toFixed(1)}</span>
                                <span class="material-symbols-outlined text-[12px]" data-icon="star" style="font-variation-settings: 'FILL' 1;">star</span>
                            </div>
                        </div>
                        <div class="mt-4 flex items-center gap-2">
                            <span class="material-symbols-outlined text-[16px] text-green-600">verified</span>
                            <span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Based on your history</span>
                        </div>
                    </div>
                </div>
                `;
            });
        }
    } catch(e) {
        console.error("Failed to load analytics", e);
    }

    // Fetch and Render Restaurants List
    try {
        const restData = await fetchRestaurants();
        const restContainer = container.querySelector('#home-restaurants-list');
        if (restContainer && restData.restaurants) {
            restContainer.innerHTML = '';
            restData.restaurants.forEach(rest => {
                const imgId = Math.abs(rest.name.split('').reduce((a,b)=>{a=((a<<5)-a)+b.charCodeAt(0);return a&a},0)) % 10 + 1;
                const restImgUrl = `https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&h=300&fit=crop&q=80`; // Placeholder for now
                
                const card = document.createElement('div');
                card.className = "bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100 mb-6 cursor-pointer active:scale-[0.98] transition-transform";
                card.onclick = () => navigate('menu', { cuisine: rest.cuisine, restaurant: rest.name });
                
                card.innerHTML = `
                    <div class="relative h-48">
                        <img src="${restImgUrl}" class="w-full h-full object-cover" alt="${rest.name}">
                        <div class="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm px-2 py-1 rounded-lg flex items-center gap-1 shadow-sm">
                            <span class="text-xs font-bold">${rest.rating}</span>
                            <span class="material-symbols-outlined text-[12px] text-yellow-500" style="font-variation-settings: 'FILL' 1;">star</span>
                        </div>
                    </div>
                    <div class="p-4">
                        <div class="flex justify-between items-start">
                            <div>
                                <h3 class="font-headline font-bold text-gray-900">${rest.name}</h3>
                                <p class="text-xs text-gray-500 mt-1">${rest.cuisine} • ${rest.deliveryTime}</p>
                            </div>
                        </div>
                    </div>
                `;
                restContainer.appendChild(card);
            });
        }
    } catch(e) {
        console.error("Failed to load restaurants", e);
    }
}

// 2. MENU VIEW
async function renderMenuView() {
    const container = document.getElementById('view-menu');
    
    if (!container.innerHTML.trim()) {
        const html = await fetchTemplate('menu');
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        container.innerHTML = doc.body.innerHTML;
        
        // Fix back button
        const backBtn = container.querySelector('header button');
        if (backBtn) backBtn.addEventListener('click', () => navigate('home'));
        
        // Fix Cart button in Sticky Footer
        const cartBanner = container.querySelector('footer');
        if (cartBanner) {
            cartBanner.id = "menu-cart-banner";
            cartBanner.style.cursor = "pointer";
            cartBanner.addEventListener('click', () => navigate('cart'));
        }
    }

    // Update Header with Restaurant Context
    const headerTitle = container.querySelector('header h1') || container.querySelector('header span.font-bold');
    if (headerTitle) {
        headerTitle.textContent = STATE.currentRestaurant;
    }
    const headerSub = container.querySelector('header p') || container.querySelector('header span.text-xs');
    if (headerSub) {
        headerSub.textContent = `${STATE.currentCuisine} • 25-35 min`;
    }

    // Update Search Placeholder
    const searchPlaceholder = container.querySelector('header .body-md') || container.querySelector('header span.text-on-surface-variant');
    if (searchPlaceholder) {
        searchPlaceholder.textContent = `Search in ${STATE.currentRestaurant}`;
    }
    
    // Fetch and Render Dynamic Menu Items
    const data = await fetchMenuData();
    const mainSection = container.querySelector('main');
    
    // Clear out the pre-existing static lists inside <main>
    mainSection.innerHTML = '';
    
    // Render By Category
    Object.entries(data.menu).forEach(([category, items]) => {
        const section = document.createElement('section');
        section.className = "mb-8";
        
        const header = document.createElement('h2');
        header.className = "font-headline text-title-md font-bold mb-6 flex items-center gap-2";
        header.innerHTML = `${category} <span class="h-1 w-12 bg-primary rounded-full"></span>`;
        section.appendChild(header);
        
        const listContainer = document.createElement('div');
        listContainer.className = "space-y-4";
        
        items.forEach(item => {
            const card = document.createElement('div');
            card.className = "bg-white p-4 rounded-xl flex justify-between gap-4 border border-gray-100 shadow-sm";
            
            const iconHtml = item.isVeg 
                ? `<div class="veg-icon mb-2"><div class="veg-dot"></div></div>`
                : `<div class="nonveg-icon mb-2"><div class="nonveg-dot"></div></div>`;
                
            // Generate a deterministic image URL based on name hash if we had them, 
            // but for simplicity, use a generic placeholder colored box or missing img icon
            const imgName = item.name.replace(/\s/g, '_') + '.png';
            const productImgUrl = `/product-images/${imgName}`;
            const avatarFallback = `https://ui-avatars.com/api/?name=${item.name.replace(/\s/g, '+')}&background=fceae9&color=E23744`;
            
            // Check if item is in cart to show quantity controls
            const cartItem = STATE.cart.find(i => i.name === item.name);
            const quantity = cartItem ? cartItem.quantity : 0;
            
            let actionButtonHtml = "";
            if (quantity > 0) {
                actionButtonHtml = `
                    <div class="absolute -bottom-3 left-1/2 -translate-x-1/2 flex items-center bg-primary border border-primary rounded-lg overflow-hidden shadow-sm">
                        <button class="px-2 py-1 text-white hover:bg-red-700 transition-colors" onclick="updateCartItemFromMenu('${item.name}', -1)">
                            <span class="material-symbols-outlined text-[16px]">remove</span>
                        </button>
                        <span class="px-2 text-white font-bold text-xs min-w-[20px] text-center">${quantity}</span>
                        <button class="px-2 py-1 text-white hover:bg-red-700 transition-colors" onclick="updateCartItemFromMenu('${item.name}', 1)">
                            <span class="material-symbols-outlined text-[16px]">add</span>
                        </button>
                    </div>
                `;
            } else {
                actionButtonHtml = `
                    <button class="add-to-cart-btn absolute -bottom-3 left-1/2 -translate-x-1/2 bg-white text-primary border border-gray-200 font-bold px-6 py-1 rounded-lg shadow-sm text-sm"
                        data-name="${item.name}" data-price="${item.price}" data-category="${category}">
                        ADD
                    </button>
                `;
            }

            card.innerHTML = `
                <div class="flex-1">
                    ${iconHtml}
                    <h3 class="font-headline text-on-surface font-semibold text-base">${item.name}</h3>
                    <p class="text-on-surface font-medium mt-1">₹${item.price}</p>
                    <p class="text-gray-500 text-sm mt-2 line-clamp-2">${item.description}</p>
                </div>
                <div class="relative w-28 h-28 shrink-0">
                    <img src="${productImgUrl}" onerror="this.src='${avatarFallback}'" class="w-full h-full object-cover rounded-xl" alt="${item.name}">
                    ${actionButtonHtml}
                </div>
            `;
            listContainer.appendChild(card);
        });
        
        section.appendChild(listContainer);
        mainSection.appendChild(section);
    });
    
    // Bind Add Buttons
    container.querySelectorAll('.add-to-cart-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // prevent card click
            const name = btn.getAttribute('data-name');
            const price = parseFloat(btn.getAttribute('data-price'));
            const category = btn.getAttribute('data-category');
            
            addToCart({ name, price, category, quantity: 1 });
        });
    });
}

// 3. CART VIEW
async function renderCartView() {
    const container = document.getElementById('view-cart');
    
    if (!container.innerHTML.trim()) {
        const html = await fetchTemplate('cart');
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        container.innerHTML = doc.body.innerHTML;
        
        const backBtn = container.querySelector('header button');
        if (backBtn) backBtn.addEventListener('click', () => navigate('menu'));
        
        // Place Order button
        const placeOrderBtn = container.querySelector('footer button');
        if (placeOrderBtn) {
            placeOrderBtn.addEventListener('click', submitOrder);
        }
    }
    
    // Render Cart Items
    const cartSection = container.querySelector('section:nth-of-type(1)');
    // keep header but clear items
    const headerHtml = `
        <div class="flex items-center justify-between mb-4">
            <h2 class="font-headline text-title-md font-bold">Your Order</h2>
            <span class="text-label-sm font-bold bg-red-100 text-primary px-2 py-1 rounded-lg">${STATE.cart.length} ITEM(S)</span>
        </div>
    `;
    
    let itemsHtml = '';
    STATE.cartTotal = 0;
    
    STATE.cart.forEach((item, index) => {
        STATE.cartTotal += item.price * item.quantity;
        itemsHtml += `
            <div class="bg-white rounded-xl p-4 flex items-center gap-4 border border-gray-100 shadow-sm mb-3">
                <div class="flex-grow">
                    <h3 class="font-headline text-base font-semibold">${item.name}</h3>
                    <p class="text-gray-500 text-sm">₹${item.price}</p>
                </div>
                <div class="flex items-center bg-primary border border-primary rounded-lg overflow-hidden shadow-sm">
                    <button class="px-2 py-1 text-white hover:bg-red-700 transition-colors" onclick="updateCartItem(${index}, -1)">
                        <span class="material-symbols-outlined text-[18px]">remove</span>
                    </button>
                    <span class="px-2 text-white font-bold text-sm min-w-[20px] text-center">${item.quantity}</span>
                    <button class="px-2 py-1 text-white hover:bg-red-700 transition-colors" onclick="updateCartItem(${index}, 1)">
                        <span class="material-symbols-outlined text-[18px]">add</span>
                    </button>
                </div>
            </div>
        `;
    });
    
    if(STATE.cart.length === 0) {
        itemsHtml = `<div class="p-6 text-center text-gray-500">Your cart is empty. Please add items from the menu.</div>`;
    }
    
    cartSection.innerHTML = headerHtml + itemsHtml;
    
    // Render Bill Details
    const billTotal = STATE.cartTotal;
    const taxes = billTotal * 0.05;
    const delivery = billTotal > 0 ? 35 : 0;
    const grandTotal = billTotal + taxes + delivery;
    
    const billSection = container.querySelector('section:nth-of-type(3) .bg-surface-container-low');
    if (billSection) {
        billSection.innerHTML = `
            <div class="flex justify-between items-center text-sm mb-2">
                <span class="text-gray-600">Item Total</span>
                <span class="font-medium">₹${billTotal.toFixed(2)}</span>
            </div>
            <div class="flex justify-between items-center text-sm mb-2">
                <span class="text-gray-600">Delivery Fee</span>
                <span class="font-medium">₹${delivery.toFixed(2)}</span>
            </div>
            <div class="flex justify-between items-center text-sm mb-3">
                <span class="text-gray-600">Taxes & Charges</span>
                <span class="font-medium">₹${taxes.toFixed(2)}</span>
            </div>
            <div class="pt-3 border-t border-gray-200 flex justify-between items-center">
                <span class="font-bold">Grand Total</span>
                <span class="font-bold text-lg">₹${grandTotal.toFixed(2)}</span>
            </div>
        `;
    }
    
    // Update sticky footer
    const footerTotal = container.querySelector('footer p.text-headline-sm');
    if(footerTotal) footerTotal.textContent = `₹${grandTotal.toFixed(2)}`;
    
    // Make sure rail is cleared / skeletonized every time we hit the page
    const railContainer = document.querySelector('#view-cart .hide-scrollbar');
    if (railContainer) {
        railContainer.innerHTML = ''; 
        for(let i=0; i<3; i++){
            railContainer.innerHTML += `
             <div class="flex-shrink-0 w-36 bg-white rounded-xl overflow-hidden shadow-sm border border-gray-100">
                <div class="h-24 w-full skeleton"></div>
                <div class="p-3 bg-white">
                    <div class="h-4 w-24 skeleton mb-2"></div>
                    <div class="h-3 w-10 skeleton mb-4"></div>
                    <div class="h-8 w-full skeleton rounded-lg"></div>
                </div>
            </div>`;
        }
    }
}


// --- DOMAIN LOGIC ---

function addToCart(item) {
    const existing = STATE.cart.find(i => i.name === item.name);
    if (existing) {
        existing.quantity += 1;
    } else {
        STATE.cart.push(item);
    }
    
    updateMenuCartBanner();
    
    // Refresh menu if we are in it
    if (STATE.currentView === 'menu') {
        renderMenuView();
    }
}

// Global helper for menu quantity buttons
window.updateCartItemFromMenu = function(name, delta) {
    const itemIndex = STATE.cart.findIndex(i => i.name === name);
    if (itemIndex !== -1) {
        updateCartItem(itemIndex, delta);
    } else if (delta > 0) {
        // This shouldn't happen with the current UI logic but for safety:
        // Find item in menu data to get price and category
        let found = null;
        Object.entries(STATE.menuData.menu).forEach(([cat, items]) => {
            const item = items.find(i => i.name === name);
            if (item) found = { ...item, category: cat };
        });
        if (found) addToCart({ ...found, quantity: 1 });
    }
}

function updateCartItem(index, delta) {
    STATE.cart[index].quantity += delta;
    if (STATE.cart[index].quantity <= 0) {
        STATE.cart.splice(index, 1);
    }
    // Re-render
    if (STATE.currentView === 'cart') {
        renderCartView();
        fetchSmartAddons();
    } else if (STATE.currentView === 'menu') {
        renderMenuView();
    }
    updateMenuCartBanner();
}

function updateMenuCartBanner() {
    const banner = document.getElementById('menu-cart-banner');
    if (!banner) return;
    
    if (STATE.cart.length === 0) {
        banner.style.display = 'none';
        return;
    }
    
    banner.style.display = 'block';
    
    const count = STATE.cart.reduce((sum, item) => sum + item.quantity, 0);
    const total = STATE.cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    
    banner.querySelector('p').textContent = `${count} Item${count > 1 ? 's' : ''} | ₹${total.toFixed(2)}`;
}

// THE CORE RECOMMENDATION FEATURE
async function fetchSmartAddons() {
    if (STATE.cart.length === 0) return;
    
    // Map internal cart state to API request format
    const payload = {
        user_id: STATE.userId,
        restaurant_cuisine: STATE.currentCuisine,
        restaurant_name: STATE.currentRestaurant,
        city: 'Delhi-NCR',
        cart: STATE.cart.map(c => ({
            name: c.name,
            category: c.category,
            quantity: c.quantity,
            unit_price: c.price
        }))
    };
    
    try {
        const res = await fetch('/cart/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        const recommendations = data.recommendations || [];
        
        // Inject into the Horizontal Rail
        const railContainer = document.querySelector('#view-cart .hide-scrollbar');
        if (!railContainer) return;
        
        railContainer.innerHTML = '';
        
        recommendations.slice(0, 5).forEach(rec => {
            const imgName = rec.name.replace(/\s/g, '_') + '.png';
            const productImgUrl = `/product-images/${imgName}`;
            const avatarFallback = `https://ui-avatars.com/api/?name=${rec.name.replace(/\s/g, '+')}&background=random`;
            
            railContainer.innerHTML += `
            <div class="flex-shrink-0 w-36 bg-white rounded-xl overflow-hidden shadow-sm border border-gray-100">
                <div class="h-24 w-full">
                    <img src="${productImgUrl}" onerror="this.src='${avatarFallback}'" class="w-full h-full object-cover" alt="${rec.name}">
                </div>
                <div class="p-3">
                    <h4 class="text-label-sm font-bold truncate" title="${rec.name}">${rec.name}</h4>
                    <p class="text-gray-500 text-xs mb-2">₹${rec.price.toFixed(2)}</p>
                    <button onclick="addRecommendationToCart('${rec.name}', ${rec.price}, '${rec.category}')"
                        class="w-full py-1 border border-primary text-primary text-xs font-bold rounded-lg hover:bg-red-50 transition-colors">
                        ADD
                    </button>
                </div>
            </div>
            `;
        });
        
    } catch (e) {
        console.error("Failed to fetch smart addons", e);
    }
}

// Expose globally for the onclick handler in string interpolation
window.addRecommendationToCart = function(name, price, category) {
    addToCart({ name, price, category, quantity: 1 });
    renderCartView();
    fetchSmartAddons(); // Refetch contextually based on the *new* cart
}

async function submitOrder() {
    if (STATE.cart.length === 0) return alert("Cart is empty!");
    
    const payload = {
        user_id: STATE.userId,
        cart: STATE.cart.map(c => ({
            name: c.name,
            category: c.category,
            quantity: c.quantity,
            unit_price: c.price
        }))
    };
    
    try {
        const placeOrderBtn = document.querySelector('#view-cart footer button');
        const origText = placeOrderBtn.innerHTML;
        placeOrderBtn.innerHTML = "Placing...";
        
        await fetch('/checkout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        // Clear cart
        STATE.cart = [];
        
        // Flash success and navigate home
        placeOrderBtn.innerHTML = "Success!";
        placeOrderBtn.classList.replace('bg-primary', 'bg-green-500');
        
        setTimeout(() => {
            navigate('home');
        }, 1500);
        
    } catch (e) {
        alert("Checkout failed");
        console.error(e);
    }
}


// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
    // Start at Home
    navigate('home');
    initSimulator();
});

// --- SIMULATOR LOGIC ---

function toggleSimulator() {
    const panel = document.getElementById('simulator-panel');
    if (panel.classList.contains('translate-y-full')) {
        panel.classList.remove('translate-y-full');
    } else {
        panel.classList.add('translate-y-full');
    }
}

function showToast() {
    const toast = document.getElementById('simulator-toast');
    toast.classList.remove('opacity-0', '-translate-y-4');
    toast.classList.add('opacity-100', 'translate-y-0');
    
    setTimeout(() => {
        toast.classList.remove('opacity-100', 'translate-y-0');
        toast.classList.add('opacity-0', '-translate-y-4');
    }, 3000);
}

function initSimulator() {
    const presetSelect = document.getElementById('sim-preset-select');
    const aovSlider = document.getElementById('sim-aov-slider');
    const orderSlider = document.getElementById('sim-order-slider');
    const aovDisplay = document.getElementById('sim-aov-display');
    const orderDisplay = document.getElementById('sim-order-display');

    // Update displays when sliders move manually
    aovSlider.addEventListener('input', (e) => {
        aovDisplay.textContent = `₹${e.target.value}`;
    });
    orderSlider.addEventListener('input', (e) => {
        orderDisplay.textContent = e.target.value;
    });

    // Auto-update sliders when preset changes
    presetSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === "The Health Nut") {
            aovSlider.value = 500;
            orderSlider.value = 15;
        } else if (val === "The Budget Student") {
            aovSlider.value = 200;
            orderSlider.value = 5;
        } else if (val === "The Family/Bulk Orderer") {
            aovSlider.value = 1500;
            orderSlider.value = 8;
        } else if (val === "The Brand New User") {
            aovSlider.value = 0;
            orderSlider.value = 0;
        }
        
        // Trigger generic input event to update displays
        aovSlider.dispatchEvent(new Event('input'));
        orderSlider.dispatchEvent(new Event('input'));
    });
    
    // Initialize default display
    presetSelect.dispatchEvent(new Event('change'));
}

async function applyPersona() {
    const presetSelect = document.getElementById('sim-preset-select');
    const aovSlider = document.getElementById('sim-aov-slider');
    const orderSlider = document.getElementById('sim-order-slider');
    
    const payload = {
        persona_type: presetSelect.value,
        mean_aov: parseFloat(aovSlider.value),
        order_count: parseInt(orderSlider.value, 10)
    };
    
    try {
        const res = await fetch(`/user/${STATE.userId}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error("Simulation failed");
        
        // Close panel and show success toast
        toggleSimulator();
        showToast();
        
        // Force refresh UI based on current view to reflect new persona vectors
        if (STATE.currentView === 'home') {
            const container = document.getElementById('view-home');
            // Hard wipe the recommendations to force the skeleton/re-render cleanly
            const recContainer = container.querySelector('#home-recommendations');
            if(recContainer) recContainer.innerHTML = '';
            await renderHomeView();
        } else if (STATE.currentView === 'cart') {
            await fetchSmartAddons();
        }
        
    } catch(e) {
        console.error(e);
        alert("Failed to apply persona override");
    }
}
