let tasks = [];

function addTask() {
    let taskInput = document.getElementById("taskInput");
    let taskTime = document.getElementById("taskTime");
    let taskText = taskInput.value.trim();
    let taskDue = taskTime.value;

    if (taskText === "" || taskDue === "") {
        alert("Please enter a task and a deadline!");
        return;
    }

    let task = { text: taskText, due: new Date(taskDue), id: Date.now() };
    tasks.push(task);
    renderTasks();
    scheduleReminder(task);
    
    taskInput.value = "";
    taskTime.value = "";
}

function renderTasks() {
    let taskList = document.getElementById("taskList");
    taskList.innerHTML = "";
    tasks.forEach(task => {
        let li = document.createElement("li");
        li.innerHTML = `${task.text} (Due: ${task.due.toLocaleString()}) 
                        <button onclick="removeTask(${task.id})">X</button>`;
        taskList.appendChild(li);
    });
}

function removeTask(id) {
    tasks = tasks.filter(task => task.id !== id);
    renderTasks();
}

function scheduleReminder(task) {
    let timeDiff = task.due - new Date();
    if (timeDiff > 0) {
        setTimeout(() => {
            alert(`Reminder: ${task.text} is due now!`);
        }, timeDiff);
    }
}

renderTasks();
