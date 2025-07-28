// different config.js based on the link
let hash = window.location.hash.replace('#', '');
console.log("Hash from URL: ", hash);

// Get references to DOM elements
const imageContainer = document.getElementById("image-container");
const cutTitle = document.getElementById("cut-title");
const modeList = document.getElementById("mode-list");
const imageData = imageMap[hash] || { images: [], title: 'No plots found' };
let imagesPerRow = imageData.imagesPerRow || 3; // Default to 2 if not set
let currentImageSubset = 0; 

// Function to create buttons 
if (imageData.buttons) {
    const li = document.createElement("li");
    const mainNavText = document.createTextNode("Select: ");
    li.appendChild(mainNavText);
    // const numSubsets = imageData.button.length;
    imageData.buttons.forEach((button, index) => {

        const a = document.createElement("a");
        a.href = "#";
        a.className = "mode-btn sub-mode-btn";
        a.textContent = button;

        a.addEventListener('click', (e) => {
            e.preventDefault();
            currentImageSubset = index;
            document.querySelectorAll('.sub-mode-btn').forEach(b => b.classList.remove('active'));
            a.classList.add('active');
            updateImages();
        })
        a.style.margin = "0 20px"; // Add some spacing
        li.appendChild(a);
    modeList.appendChild(li);
    });

    setTimeout(() => {
        const firstBtn = document.querySelector('.sub-mode-btn');
        if (firstBtn) firstBtn.classList.add('active');
    }, 0);
}


// Function to update images based on selected cut
function updateImages() {
    imageContainer.innerHTML = "";
    cutTitle.textContent = imageData.title;

    imageContainer.style.display = "grid";
    imageContainer.style.gridTemplateColumns = `repeat(${imagesPerRow}, 1fr)`;

    let images = imageData.images;
    if (imageData.buttons) {
        images = images[currentImageSubset] || [];
    }

    images = images.map(file => `${imageData.path}/${file}`);
    images.forEach((img) => {
        const container = document.createElement('div');
        container.className = 'image-container';

        const imgElement = document.createElement('img');
        imgElement.src = img;
        imgElement.alt = img.split('/').pop();
        imgElement.onclick = () => openModal(img); // click to zoom

        const filename = document.createElement('p');
        filename.className = 'filename';
        filename.textContent = img.split('/').pop();

        container.appendChild(imgElement);
        container.appendChild(filename);
        imageContainer.appendChild(container);

    })
}

updateImages();

document.querySelectorAll('.cut-btn').forEach(btn => {
    if (parseInt(btn.textContent) === imagesPerRow) {
        btn.classList.add('active');
    } else {
        btn.classList.remove('active');
    }
});

// Change the images per row based on the cut selected 
document.querySelectorAll('.cut-btn').forEach(btn => {

    btn.addEventListener('click', (e) => {
        e.preventDefault();
        imagesPerRow = parseInt(btn.textContent);
        document.querySelectorAll('.cut-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateImages();
    });
});

// // Modal Function
function openModal(src) {
    const modal = document.getElementById("image-modal");
    document.getElementById("modal-img").src = src;
    modal.classList.add("show"); // Only add the class now
}  

// Close Modal When Clicking Outside Image
document.querySelector(".close").onclick = function () {
    document.getElementById("image-modal").classList.remove("show");
};

document.getElementById("image-modal").onclick = function (event) {
    if (event.target === this) {
        this.classList.remove("show");
    }
};
