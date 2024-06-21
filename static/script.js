document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("loginForm").addEventListener("submit", function(event) {
        event.preventDefault();

        var username = document.getElementById("loginUsername").value;
        var password = document.getElementById("password-field").value;

        // Make the POST request to the server
        fetch('/admin_login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'username': username,
                'password': password
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = "dashboard.html";
            } else {
                alert("Invalid username or password");
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});

function adminloginpage(){
    window.location.href = "admin.html";
}

function userloginpage() {
    window.location.href = "user.html";
}

let isVerificationActive = false;
let videoElement = document.getElementById('videoElement');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');

// Function to stop verification when button is clicked
function stopVerification() {
    isVerificationActive = false;
    // Send a POST request to stop the verification process
    fetch('/stop_verification', {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to stop verification');
        }
        // Verification process stopped successfully
        // You might want to add some UI feedback here if needed
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while stopping the verification process.');
    });
}

function classm(){
    window.location.href = "classm.html";
}

function studentm(){
    window.location.href = "dashboard.html";
}

// Search functionality
document.getElementById('searchInput').addEventListener('input', function() {
    const searchValue = this.value.toLowerCase().trim();
    const rows = document.querySelectorAll('#dataTable tbody tr');

    rows.forEach(row => {
        const id = row.querySelector('th').textContent.toLowerCase();
        const name = row.querySelector('td:nth-child(3)').textContent.toLowerCase();
        if (id.includes(searchValue) || name.includes(searchValue)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
});
