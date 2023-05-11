import './style.css'
import VectorSource from './node_modules/ol/source/Vector.js';
import VectorLayer from './node_modules/ol/layer/Vector.js';
import ImageLayer from './node_modules/ol/layer/Image.js';
import ImageStatic from './node_modules/ol/source/ImageStatic.js';

import WKT from './node_modules/ol/format/WKT.js';
import DragAndDrop from './node_modules/ol/interaction/DragAndDrop.js';
import Map from './node_modules/ol/Map.js';
import View from './node_modules/ol/View.js';
import * as olProj from 'ol/proj';
import GeoJSON from './node_modules/ol/format/GeoJSON.js';
import TileLayer from './node_modules/ol/layer/Tile.js';
import XYZ from './node_modules/ol/source/XYZ.js';
import OSM from './node_modules/ol/source/OSM.js';
import {ScaleLine, defaults as defaultControls} from './node_modules/ol/control.js';
import {createBox} from './node_modules/ol/interaction/Draw.js';
import {
    getPointResolution,
    get as getProjection,
    transform,
} from './node_modules/ol/proj.js';
import MousePosition from './node_modules/ol/control/MousePosition.js';
import {createStringXY} from './node_modules/ol/coordinate.js';
import {
    Draw, 
    Modify, 
    Snap, 
} from './node_modules/ol/interaction.js';
import {
    Circle as CircleStyle,
    Fill,
    Stroke,
    Style,
    Text
} from './node_modules/ol/style.js';
import UnaryUnionOp from "./node_modules/jsts/org/locationtech/jts/operation/union/UnaryUnionOp.js";
import Feature from "./node_modules/ol/Feature.js";

import MultiPolygon from './node_modules/ol/geom/MultiPolygon.js';
import Polygon from './node_modules/ol/geom/Polygon.js';

var zoneOfInterest = [];
var varwkt;
var varbound;
const key = '4Z4vZj5CICocrdP4mCFb';
const viewProjSelect = "EPSG:3857";
const sent = 'b351739d-40a8-4e8a-b943-701ef8249e08';
const layer = 'IW_VV_DB';
const scaleControl = new ScaleLine({
    units: 'metric',
    bar: true,
    dpi: 250,
    steps: 4,
    text: true,
    minWidth: 50,
    maxWidth: 100,
});

const source = new VectorSource({wrapX: false});
const vector = new VectorLayer({
    source: source,
    style: {
        'fill-color': 'rgba(0, 0, 0, 0.2)',
        'stroke-color': '#ffcc33',
        'stroke-width': 2,
        'circle-radius': 5,
        'circle-fill-color': '#ffcc33',
    },
});

const layer_sat = new TileLayer({
    source: new XYZ({
        url:
            'https://api.maptiler.com/tiles/satellite/{z}/{x}/{y}.jpg?key=' + key,
        maxZoom: 15,
    }),
});
const layer_schema = new TileLayer({
    source: new OSM({
        attributions: '',
        maxZoom: 15,
    }),
});

const mousePositionControl = new MousePosition({
    coordinateFormat: createStringXY(4),
    projection: 'EPSG:4326',
    className: 'custom-mouse-position',
    target: document.getElementById('mouse-position'),
});
const mousePositionControl2 = new MousePosition({
    coordinateFormat: createStringXY(4),
    projection: 'EPSG:3857',
    className: 'custom-mouse-position2',
    target: document.getElementById('mouse-position2'),
});

let map = new Map({
    controls: defaultControls().extend([mousePositionControl, mousePositionControl2, scaleControl]),
    layers: [layer_schema, vector],
    target: 'map',
    view: new View({
        center: [4187580.9902, 7508989.6804],
        zoom: 6,
    }),
});

const extent = getProjection('EPSG:3857').getExtent().slice();
extent[0] += extent[0];
extent[2] += extent[2];
const precisionInput = document.getElementById('precision');
precisionInput.addEventListener('change', function (event) {
    const precision_val = event.target.valueAsNumber;
    var overlay_width = 68;
    if (precision_val > 6){
        overlay_width = 78
    }
    document.getElementById("overlay").style.width = `${overlay_width}%`;
    const format = createStringXY(precision_val);
    mousePositionControl.setCoordinateFormat(format);
    mousePositionControl2.setCoordinateFormat(format);
});

let dragAndDropInteraction;
var lst = [];

function setInteraction() {
    if (dragAndDropInteraction) {
        map.removeInteraction(dragAndDropInteraction);
    }
    dragAndDropInteraction = new DragAndDrop({
        formatConstructors: [
            GeoJSON,
        ],
    });
    dragAndDropInteraction.on('addfeatures', function (event) {
        const vectorSource = new VectorSource({
            features: event.features,
        });

        lst.push(vectorSource)
        var lst_clear = [];
        for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
            if (map.getLayers().array_[i].values_['zIndex'] !== 1 || map.getLayers().array_[i].values_['zIndex'] !== 3) {
                lst_clear.push(map.getLayers().array_[i])
            }
        }

        map.setLayers([styles[styleSelector.value]]);
        map.addLayer(
            new VectorLayer({
                source: vectorSource,
                zIndex: 1
            })
        );
        map.getView().fit(vectorSource.getExtent());
    });
    map.addInteraction(dragAndDropInteraction);
}

setInteraction();

const displayFeatureInfo = function (pixel) {
    const features = [];
    map.forEachFeatureAtPixel(pixel, function (feature) {
        features.push(feature);
    });
    if (features.length > 0) {
        const info = [];
        let i, ii;
        for (i = 0, ii = features.length; i < ii; ++i) {
            info.push(features[i].get('name'));
        }
    }
};

map.on('pointermove', function (evt) {
    if (evt.dragging) {
        return;
    }
    const pixel = map.getEventPixel(evt.originalEvent);
    displayFeatureInfo(pixel);
});
map.on('click', function (evt) {
    displayFeatureInfo(evt.pixel);
});

const styleSelector = document.getElementById('style');
const styles = {layer_schema, layer_sat};

function update() { // TODO 24.03
    var remember = [];
    for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
        if (map.getLayers().array_[i].values_['zIndex'] === 0) {
            remember.push(map.getLayers().array_[i])
        }
    }

    map.setLayers([styles[styleSelector.value]]);
    for (let i = 0; i < lst.length; i++) {
        map.addLayer(
            new VectorLayer({
                source: lst[i],
            })
        );
        map.getView().fit(lst[i].getExtent());
    }
    map.addLayer(vector);

    if (remember.length !== 0) {
        map.addLayer(remember[0]);
    }
}

function clearLayers() {
    document.getElementById('clearBtnL').disabled = true;
    document.getElementById('userResD').classList.add('text-secondary');
    document.getElementById('userResD').classList.remove('text-primary');
    document.getElementById('exportBtnL').disabled = true;
    document.getElementById('userResG').classList.add('text-secondary');
    document.getElementById('userResG').classList.remove('text-primary');

    var lst = [];
    for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
        if (map.getLayers().array_[i].values_['zIndex'] !== 1 && map.getLayers().array_[i].values_['zIndex'] !== 3) {
            lst.push(map.getLayers().array_[i])
        }
    }

    map.setLayers([styles[styleSelector.value]]);
    for (let i = 1; i < lst.length; i++) {
        map.addLayer(lst[i]);
    }
}

styleSelector.addEventListener('change', update);

document.getElementById('undo').addEventListener('click', function () {
    draw.removeLastPoint();
});
document.getElementById('abort').addEventListener('click', function () {
    draw.abortDrawing();
    document.getElementById('clear').disabled = true;
    document.getElementById('exportBtn').disabled = true;
    document.getElementById('undo').disabled = true;
    document.getElementById('abort').disabled = true;

    document.getElementById('userShp').classList.add('text-secondary');
    document.getElementById('userShp').classList.remove('text-primary');
});
document.getElementById('clear').addEventListener('click', function () {
    document.getElementById('clear').disabled = true;
    document.getElementById('exportBtn').disabled = true;
    document.getElementById('undo').disabled = true;
    document.getElementById('abort').disabled = true;

    document.getElementById('userShp').classList.add('text-secondary');
    document.getElementById('userShp').classList.remove('text-primary');
    source.clear();
});
document.getElementById("exportBtn").addEventListener('click', function () {
    var features = source.getFeatures();
    var json = new GeoJSON().writeFeatures(features, {
        dataProjection: 'EPSG:3857', featureProjection: 'EPSG:3857'
    });
    function download(content, fileName, contentType) {
        var a = document.createElement("a");
        var file = new Blob([content], {type: contentType});
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
        a.remove();
    }
    download(json, 'feature_export_3857.json', 'application/json');

    var json = new GeoJSON().writeFeatures(features, {
            dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'
        });
        function download(content, fileName, contentType) {
            var a = document.createElement("a");
            var file = new Blob([content], {type: contentType});
            a.href = URL.createObjectURL(file);
            a.download = fileName;
            a.click();
            a.remove();
        }
        download(json, 'feature_export_4326.json', 'application/json');
});
document.getElementById("exportBtnL").addEventListener('click', function () {
    document.getElementById('exportBtnL').disabled = true;
    document.getElementById('userResG').classList.add('text-secondary');
    document.getElementById('userResG').classList.remove('text-primary');
    var lst = [];

    for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
        if (map.getLayers().array_[i].values_['zIndex'] === 1 || map.getLayers().array_[i].values_['zIndex'] === 3) {
            lst.push(map.getLayers().array_[i])
        }
    }

    var f_lst = []
    var feat = []

    for (let i = 0; i < lst.length; i++) {
        f_lst.push(lst[i].getSource().getFeatures());
    }
    for (let i = 0; i < f_lst.length; i++) {
        for (let j = 0; j < f_lst[i].length; j++) {
            feat.push(f_lst[i][j]);
        }
    }
    var json = new GeoJSON().writeFeatures(feat, {
        dataProjection: 'EPSG:3857', featureProjection: 'EPSG:3857'
    });
    function download_l(content, fileName, contentType) {
        var a = document.createElement("a");
        var file = new Blob([content], {type: contentType});
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
        a.remove();
    }
    download_l(json, 'layer_export_3857.json', 'application/json');

    var json = new GeoJSON().writeFeatures(feat, {
            dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'
        });
        function download_l(content, fileName, contentType) {
            var a = document.createElement("a");
            var file = new Blob([content], {type: contentType});
            a.href = URL.createObjectURL(file);
            a.download = fileName;
            a.click();
            a.remove();
        }
        download_l(json, 'layer_export_4326.json', 'application/json');
});
document.getElementById("clearBtnL").addEventListener('click', function () {
    clearLayers();
});
document.getElementById("clearBtnI").addEventListener('click', function () {
    var lst = [];
    for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
        if (map.getLayers().array_[i].values_['zIndex'] !== 2 && map.getLayers().array_[i].values_['zIndex'] !== 0) {
            lst.push(map.getLayers().array_[i])
        }
    }
    map.setLayers(lst);
    document.getElementById('clearBtnI').disabled = true;
    document.getElementById('userImg').classList.add('text-secondary');
    document.getElementById('userImg').classList.remove('text-primary');
});
let my_str;


const modifyStyle = new Style({
    image: new CircleStyle({
        radius: 5,
        stroke: new Stroke({
            color: 'rgba(0, 0, 0, 0.5)',
        }),
        fill: new Fill({
            color: 'rgba(0, 0, 0, 0.4)',
        }),
    }),
    text: new Text({
        text: 'Понятите, чтобы измнить',
        font: '12px Calibri,sans-serif',
        fill: new Fill({
            color: 'rgba(255, 255, 255, 1)',
        }),
        backgroundFill: new Fill({
            color: 'rgba(0, 0, 0, 0.5)',
        }),
        padding: [2, 2, 2, 2],
        textAlign: 'left',
        offsetX: 15,
    }),
});

const modify = new Modify({source: source, style: modifyStyle});
map.addInteraction(modify);
let draw, snap;

// ----------------------------------------------------------------------
const server_url = 'http://localhost:8000'

function showRegisterPage() {
    $('#register_page').show()
    $('#login_page').hide()
    $('#workbench_page').hide()
    $('#account_page').hide()
}

function showLoginPage() {
    $('#register_page').hide()
    $('#login_page').show()
    $('#workbench_page').hide()
    $('#account_page').hide()
}

function showMainPage() {
    $('#register_page').hide()
    $('#login_page').hide()
    $('#workbench_page').show()
    $('#account_page').hide()
}

function showAccountPage() {
    $('#register_page').hide()
    $('#login_page').hide()
    $('#workbench_page').hide()
    $('#account_page').show()
    
    document.getElementById("login").style.borderColor = '';
    document.getElementById("currPwd").style.borderColor = '';
    document.getElementById("newPwd").style.borderColor = '';

    document.getElementById("loginError").visibility = "hidden";
    document.getElementById("currPwdError").visibility = "hidden";
    document.getElementById("newPwdError").visibility = "hidden";
}

function clearLocalStorage() {
    localStorage.clear()
}

function clearLoginPage() {
    document.getElementById("logLogin").value = ''
    document.getElementById("logPwd").value = ''
    document.getElementById("logLogin").style.borderColor = ''
    document.getElementById("logPwd").style.borderColor = ''
    
    const logError = document.getElementById("logLoginError");
    const pwdError = document.getElementById("logPwdError");
    logError.style.visibility = "hidden";
    pwdError.style.visibility = "hidden";

    document.getElementById("logLoginError").value = '';
    document.getElementById("logPwdError").value = '';
}

function validateAccForm(){
    const username = document.getElementById("login");
    const password = document.getElementById("currPwd");
    const logError = document.getElementById("loginError");
    const pwdError = document.getElementById("currPwdError");
    
    const new_password = document.getElementById("newPwd");
    const new_pwdError = document.getElementById("newPwdError");

    var usernameInvalid = false;
    var passwordInvalid = false;
    var new_passwordInvalid = false;

    if (new_password.value !== '' && new_password.value.length < 8){
        new_password.style.borderColor = 'red';
        new_pwdError.style.visibility = "visible";
        new_passwordInvalid = true;
        new_pwdError.textContent = "Новый пароль должен быть не менее 8 символов";
    } else {
        if (new_password.value !== ''){
            new_password.style.borderColor = 'green';
        }
        new_pwdError.style.visibility = "hidden";
    }

    if (username.value === '') {
        username.style.borderColor = 'red';
        usernameInvalid = true;
        logError.style.visibility = "visible";
        logError.textContent = "Поле 'Логин' не должно быть пустым";
    } else {
        username.style.borderColor = 'green';
        logError.style.visibility = "hidden";
    }

    if (password.value === '') {
        password.style.borderColor = 'red';
        passwordInvalid = true;
        pwdError.style.visibility = "visible";
        pwdError.textContent = "Поле 'Пароль' не должно быть пустым";
    } else {
        password.style.borderColor = 'green';
        pwdError.style.visibility = "hidden";
    }

    return !(usernameInvalid || passwordInvalid || new_passwordInvalid);
}

function validateLoginForm() {
    const username = document.getElementById("logLogin");
    const password = document.getElementById("logPwd");
    const logError = document.getElementById("logLoginError");
    const pwdError = document.getElementById("logPwdError");
    var usernameInvalid = false;
    var passwordInvalid = false;

    if (username.value === '') {
        username.style.borderColor = 'red';
        usernameInvalid = true;
        logError.style.visibility = "visible";
        logError.textContent = "Поле 'Логин' не должно быть пустым"
    } else {
        username.style.borderColor = 'green';
        logError.style.visibility = "hidden";
    }

    if (password.value === '') {
        password.style.borderColor = 'red';
        passwordInvalid = true;
        pwdError.style.visibility = "visible";
        pwdError.textContent = "Поле 'Пароль' не должно быть пустым";
    } else {
        password.style.borderColor = 'green';
        pwdError.style.visibility = "hidden";
    }

    return !(usernameInvalid || passwordInvalid);
}

function validateRegForm(){
    const username = document.getElementById("regLogin");
    const password = document.getElementById("regPwd");
    const logError = document.getElementById("regLoginError");
    const pwdError = document.getElementById("regPwdError");
    var usernameInvalid = false;
    var passwordInvalid = false;

    if (username.value === '') {
        username.style.borderColor = 'red';
        usernameInvalid = true;
        logError.style.visibility = "visible";
        logError.textContent = "Поле 'Логин' не должно быть пустым";
    } else {
        username.style.borderColor = 'green';
        logError.style.visibility = "hidden";
    }

    if (password.value === '') {
        password.style.borderColor = 'red';
        passwordInvalid = true;
        pwdError.style.visibility = "visible";
        pwdError.textContent = "Поле 'Пароль' не должно быть пустым";
    } else {
        password.style.borderColor = 'green';
        pwdError.style.visibility = "hidden";
    }

    return !(usernameInvalid || passwordInvalid);
}

document.getElementById("logOut").addEventListener('click', function () {
    clearLocalStorage();
    showLoginPage();
});

document.getElementById("to_workbench_page").addEventListener('click', function () {
    showMainPage();
});

document.getElementById("to_acc_page").addEventListener('click', function () {
    showAccountPage();
});

document.getElementById("to_register_page_from_log").addEventListener('click', function () {
    showRegisterPage();
});

document.getElementById("to_login_page_from_reg").addEventListener('click', function () {
    showLoginPage();
});

document.getElementById("regSubmit").addEventListener('click', async function () {
    if (validateRegForm()) {
        const username = document.getElementById("regLogin").value.toString();
        const password = document.getElementById("regPwd").value.toString();
        const surname = document.getElementById("surname").value.toString();
        const name = document.getElementById("name").value.toString();
        const patronymic = document.getElementById("patronymic").value.toString();

        const url_ = `${server_url}/register/`;
        
        let response = await fetch(url_, {
            method: "POST",
            headers: {"Accept": 'application/json', "Content-type": 'application/json'},
            body: JSON.stringify({
                "user": {
                    "username": username,
                    "password": password,
                    "name": name == "" ? null : name,
                    "surname": surname == "" ? null : surname,
                    "patronymic": patronymic == "" ? null : patronymic
                }
            })
        })

        if (response.ok) {
            response = await response.json();
            clearLoginPage();
            showLoginPage();
        } else {
            console.log(response);
            response = await response.json();

            const username = document.getElementById("regLogin");
            const password = document.getElementById("regPwd");
            const logError = document.getElementById("regLoginError");
            const pwdError = document.getElementById("regPwdError");
            var usernameInvalid = false;
            var passwordInvalid = false;

            if (response["user"]["username"] !== "undefined" && 
                response["user"]["username"][0] == "user with this username already exists."){
                username.style.borderColor = 'red';
                usernameInvalid = true;
                logError.style.visibility = "visible";
                logError.textContent = "Логин занят";
            }

            if (response["user"]["password"] !== "undefined" && 
                response["user"]["password"][0] == "Ensure this field has at least 8 characters."){
                    passwordInvalid = true;
                    password.style.borderColor = 'red';
                    pwdError.style.visibility = "visible";
                    pwdError.textContent = "Пароль должен содержать 8 символов";
            }
        }
    }
});

document.getElementById("logSubmit").addEventListener('click', async function () {
    if (validateLoginForm()) {
        const username = document.getElementById("logLogin").value.toString();
        const password = document.getElementById("logPwd").value.toString();

        const url_ = `${server_url}/login/`;
        let response = await fetch(url_, {
            method: "POST",
            headers: {"Accept": 'application/json', "Content-type": 'application/json'},
            body: JSON.stringify({
                "user": {
                    "username": username,
                    "password": password
                }
            })
        })

        if (response.ok) {
            response = await response.json();
            localStorage.setItem('Token', "Bearer " + response["user"]["token"]);
            app.msg = localStorage.getItem("Token");
            clearLoginPage();
            showMainPage();
            updateOrders();
        } else {
            response = await response.json();
            const username = document.getElementById("logLogin");
            const password = document.getElementById("logPwd");
            const logError = document.getElementById("logLoginError");
            const pwdError = document.getElementById("logPwdError");
            var usernameInvalid = false;
            var passwordInvalid = false;
            
            if (response["user"]["non_field_errors"][0] == "Логин не существует"){
                username.style.borderColor = 'red';
                usernameInvalid = true;
                logError.style.visibility = "visible";;
                logError.textContent = response["user"]["non_field_errors"][0];
            }
          
            if (response["user"]["non_field_errors"][0] == "Неверный пароль"){
                passwordInvalid = true;
                password.style.borderColor = 'red';
                pwdError.style.visibility = "visible";
                pwdError.textContent = response["user"]["non_field_errors"][0];
            }
        }
    }
});

document.getElementById("newSubmit").addEventListener('click', async function () {
    if (validateAccForm()) {
        const username = document.getElementById("login").value.toString();
        const password = document.getElementById("currPwd").value.toString();
        const new_password = document.getElementById("newPwd").value.toString();
        const surname = document.getElementById("newSurname").value.toString();
        const name = document.getElementById("newName").value.toString();
        const patronymic = document.getElementById("newPatronymic").value.toString();

        const token = localStorage.getItem("Token");
        const url_ = `${server_url}/user/`;

        let response = await fetch(url_, {
            method: "PATCH",
            headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": token},
            body: JSON.stringify({
                "user": {
                    "username": username,
                    "password": password,
                    "new_password": new_password == "" ? null : new_password,
                    "name": name == "" ? null : name,
                    "surname": surname == "" ? null : surname,
                    "patronymic": patronymic == "" ? null : patronymic
                }
            })
        });

        if (response.ok) {
            response = await response.json();
            showAccountPage();
            if (new_password !== "") {
                document.getElementById("newPwd").style.borderColor = 'green';
            }
            
            if (name !== "") {
                document.getElementById("newName").style.borderColor = 'green';
            }
            
            if (surname !== "") {
                document.getElementById("newSurname").style.borderColor = 'green';
            }
            
            if (patronymic !== "") {
                document.getElementById("newPatronymic").style.borderColor = 'green';
            }
        
        } else {
            const username = document.getElementById("login");
            const password = document.getElementById("currPwd");
            const logError = document.getElementById("loginError");
            const pwdError = document.getElementById("currPwdError");

            console.log(response);
            if (response.status == 400){
                username.style.borderColor = 'red';
                logError.style.visibility = "visible";
                password.style.borderColor = 'red';
                pwdError.style.visibility = "visible";
                logError.textContent = "Заполните поля, которые нужно изменить";
                pwdError.textContent = "Заполните поля, которые нужно изменить";
            }
            
            if (response.status == 403){
                username.style.borderColor = 'red';
                logError.style.visibility = "visible";
                password.style.borderColor = 'red';
                pwdError.style.visibility = "visible";
                logError.textContent = "Неправильные данные для входа";
                pwdError.textContent = "Неправильные данные для входа";
            }
            
            if (response.status == 401){
                username.style.borderColor = 'red';
                logError.style.visibility = "visible";
                password.style.borderColor = 'red';
                pwdError.style.visibility = "visible";
                logError.textContent = "Вы не вошли в аккаунт";
                pwdError.textContent = "Вы не вошли в аккаунт";
                showLoginPage();
            }

            response = await response.json();
        }
    }
});

const interval = setInterval(function () {
    const url_ = `${server_url}/orders/`;
    const token = localStorage.getItem("Token");

    fetch(url_, {
        method: "GET",
        headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": token}
    }).then(response => response).then(response => {
        if (response.ok){
            updateOrders();
        } else {
            console.log("INTERVAL ERR:\n", response);
        }
    })
}, 1000);

document.getElementById("deleteAccount").addEventListener('click', async function () {
    const url_ = `${server_url}/me/`
    const token = localStorage.getItem("Token")

    let response = await fetch(url_, {
        method: "DELETE",
        headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": token}
    })

    if (response.ok) {
        clearLocalStorage()
        showLoginPage()
    }
});

document.getElementById("createNewOrder").addEventListener('click', function () {
    app.orders.push(1);
    document.getElementById("createNewOrder").disabled = true;
    updateOrders();
})

function updateOrders() {
    console.log("UPDATE ORDERS");
    const url_ = `${server_url}/orders/`;
    const token = localStorage.getItem("Token");

    fetch(url_, {
        method: "GET",
        headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": token}
    }).then(response => response.json()).then(data => {
        app.createdOrders = []
        app.finishedOrders = []
        data[0].forEach(cOrder => app.createdOrders.push(cOrder))
        data[1].forEach(fOrder => app.finishedOrders.push(fOrder))
    })
}

function getOrders(cOrders, fOrders) {
    console.log("GET ORDERS");
    const url_ = `${server_url}/orders/`;
    const token = localStorage.getItem("Token");

    fetch(url_, {
        method: "GET",
        headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": token}
    }).then(response => response.json()).then(data => {
        data[0].forEach(cOrder => cOrders.push(cOrder))
        data[1].forEach(fOrder => fOrders.push(fOrder))
    })
}

function sendOrder() {
    const token = localStorage.getItem("Token")
    const date1 = localStorage.getItem("date1")
    const date2 = localStorage.getItem("date2")
    const wktRepresentation = localStorage.getItem("wktRepresentation")

    const url_ = `${server_url}/order/`;
    fetch(url_, {
        method: "POST",
        headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": token},
        body: JSON.stringify({
            "order": {
                "imagery_start": date1,
                "imagery_end": date2 === "null" ? null : date2,
                "poly_wkt": wktRepresentation,
                "crs": viewProjSelect
            }
        })
    }).then(response => {
            if (!response.ok) {
                console.log(response);
            }
            updateOrders();
        }
    )

    var lst = [];
    for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
        if (map.getLayers().array_[i].values_['zIndex'] !== 0) {
            lst.push(map.getLayers().array_[i])
        }
    }

    map.setLayers(lst)
}

function setUrl(start, fin, url, wkt_rep) {
    if (start.length === 0) {
        start = "2020" + "-" + "06" + "-" + "02"
    }

    const basic = "SHOWLOGO=false&VERSION=1.3.0&MAXCC=1&WIDTH=256&HEIGHT=256&FORMAT=image/jpeg&SERVICE=WMS&REQUEST=GetMap"
    const pref = "http://services.sentinel-hub.com/ogc/wms"
    my_str = `${pref}/${sent}?CRS=${viewProjSelect}&LAYERS=${layer}&TIME=${start}/${fin}&GEOMETRY=${wkt_rep}&${basic}`
    localStorage.setItem(url, my_str)
}
function getFirstImage(startDate, finishDate) {
    var lst = [];
    for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
        if (map.getLayers().array_[i].values_['zIndex'] !== 0) {
            lst.push(map.getLayers().array_[i])
        }
    }

    map.setLayers(lst)

    var features = source.getFeatures();
    var wktRepresentation;
    var Bound;
    if (features.length === 0) {
        if (zoneOfInterest.length === 0) {
            console.log("no shapes");
        } else {
            wktRepresentation = varwkt;
            Bound = varbound;
        }
    } else {
        var format = new WKT();
        var geom = [];
        if (features.length === 1) {
            wktRepresentation = format.writeGeometry(features[0].getGeometry().clone().transform('EPSG:3857', 'EPSG:3857'));
            Bound = features[0].getGeometry().getExtent();
            zoneOfInterest = features;

            varwkt = wktRepresentation;
            varbound = Bound;
        } else {
            var olGeom = new UnaryUnionOp(features[0].getGeometry(), features[1].getGeometry());
            wktRepresentation = format.writeGeometry(olGeom._geomFact);
            Bound = olGeom._geomFact.getExtent();
        }
    }
    setUrl(startDate, finishDate, 'url', wktRepresentation)
    var img_ext = olProj.transformExtent(Bound, 'EPSG:3857', 'EPSG:3857') // EPSG:4326 3857
    var imageLayer = new ImageLayer({
        source: new ImageStatic({
            url: my_str,
            imageExtent: img_ext // east, north, west, south
        }),
        zIndex: 0
    });
    map.addLayer(imageLayer);
    source.clear();
}

function getSecondImage(startDate, sat) {
}

Vue.component('order-card-row', {
    props: ['order'],
    data: function () {
        return {
            show_1: false,
            show_2: false,
            show_3: false,
            iceSelected: false,
            col1: false,
            col2: false,
            date1: "",
            date2: ""
        }
    },
    template:
        '<div v-if="order[0] == 1" class="customCard"><p>Создание заказа</p>' +
        '<div style="all:initial;">' +
            '<div style="margin-left: 12%; margin-bottom: 3px;"><button @click="show1()" type="button" class="btn btn-primary btn-circle btn-sm"><p v-if="show_1==true">-</p><p v-if="show_1==false">+</p></button> <i @click="show1()" style="cursor: pointer;">Добавить разметку</i></div>' +
                '<div v-if="show_1 == true">' +
                    '<label class="input-group-text" for="shapeType">Geometry type:</label>\n' +
                    '              <select @change="selectArea()" class="form-select" id="shapeType">\n' +
                    '                  <option value="None">None</option>\n' +
                    '                  <option value="Polygon">Polygon</option>\n' +
                    '                  <option value="Box">Box</option>\n' +
                    '              </select>' +
                '</div>' +
            '<div style="margin-left: 12%; margin-bottom: 3px;"><button @click="show3()" type="button" class="btn btn-primary btn-circle btn-sm"><p v-if="show_3==true">-</p><p v-if="show_3==false">+</p></button> <i @click="show3()" style="cursor: pointer;">Добавить даты съемки</i></div>' +
                '<div v-if="show_3 == true"> ' +
        '           <div style="align-items: center; text-align: center;">' +
                        '<p>Start date: <input @change="pickedDate1()" type="date" id="startDatepicker"></p>\n' +
                        '<p>Finish date: <input @change="pickedDate2()" type="date" id="finishDatepicker"></p>' +
        '           </div>' +
                '</div>' +
        '<button id="showImagesButton" @click="showImages()" class="btn btn-outline-primary btn-block" data-bs-toggle="collapse" data-bs-target="#collapseExample" aria-expanded="false" aria-controls="collapseExample" style="margin-bottom: 10px;">Посмотреть</button>' +
        '<div style="align-items: center;\n' +
        '            text-align: center;"><table style="margin: 0 auto; ">' +
            '<tbody>' +
                '<tr>' +
                    '<td>' +
                        '<div v-if="show_1 == true && show_3 == true"><button @click="sendNewOrder()" class="btn btn-outline-success m-1 btn-block" style="">Отправить </button></div>' +
                        '<div v-if="!(show_1 == true && show_3 == true)"><button @click="sendNewOrder()" class="btn btn-outline-success m-1 btn-block" disabled>Отправить</button></div>' +
                    '</td>' +
                    '<td>' +
                        '<div><button @click="deleteEmptyOrder()" class="btn btn-outline-danger btn-block">Удалить</button></div>' +
                    '</td>' +
                '</tr>' +
            '</tbody>' +
        '</table></div>' +
        '</div>' +
        '</div>',
    methods: {
        pickedDate1() {
            const date1 = document.getElementById("startDatepicker").value
            this.date1 = date1
            this.col1 = date1.length > 0;
        },
        pickedDate2() {
            const date2 = document.getElementById("finishDatepicker").value
            this.date2 = date2
            this.col2 = document.getElementById("finishDatepicker").value.length > 0;
        },
        deleteEmptyOrder() {
            app.orders = []
            this.show_1 = false;
            this.show_2 = false;
            this.show_3 = false;
            this.iceSelected = false
            document.getElementById("createNewOrder").disabled = false
            
            try {
                document.getElementById('shapeType').value = 'None'
            } catch (ex){
                const _ = 0
            }
            map.removeInteraction(draw);
            map.removeInteraction(snap);

            document.getElementById('clear').disabled = true;
            document.getElementById('exportBtn').disabled = true;
            document.getElementById('undo').disabled = true;
            document.getElementById('abort').disabled = true;
            document.getElementById('userShp').classList.add('text-secondary');
            document.getElementById('userShp').classList.remove('text-primary');
            var lst = [];
            for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
                if (map.getLayers().array_[i].values_['zIndex'] !== 2 && map.getLayers().array_[i].values_['zIndex'] !== 0) {
                    lst.push(map.getLayers().array_[i])
                }
            }
            map.setLayers(lst);
            document.getElementById('clearBtnI').disabled = true;
            document.getElementById('userImg').classList.add('text-secondary');
            document.getElementById('userImg').classList.remove('text-primary');
            source.clear();

        },
        selectModel() {
            if (document.getElementById("modelType").value === "ice") {
                this.iceSelected = true
            } else {
                this.iceSelected = false
            }
        },
        selectSat() {

        },
        sendNewOrder() {
            let date1 = "";
            let date2 = "";
            if (document.getElementById("startDatepicker") != null) {
                date1 = document.getElementById("startDatepicker").value;
            }
            if (document.getElementById("finishDatepicker") != null) {
                date2 = document.getElementById("finishDatepicker").value;
            }
            var features = source.getFeatures();
            var wktRepresentation;
            var Bound;
            if (features.length === 0) {
                if (zoneOfInterest.length === 0) {
                    console.log("no shapes");
                } else {
                    wktRepresentation = varwkt;
                    Bound = varbound;
                }
            } else {
                var format = new WKT();
                if (features.length === 1) {
                    wktRepresentation = format.writeGeometry(features[0].getGeometry().clone().transform('EPSG:3857', 'EPSG:3857'));
                    Bound = features[0].getGeometry().getExtent();

                    zoneOfInterest = features;

                    varwkt = wktRepresentation;
                    varbound = Bound;
                } else {
                    var olGeom = new UnaryUnionOp(features[0].getGeometry(), features[1].getGeometry());
                    wktRepresentation = format.writeGeometry(olGeom._geomFact);
                    Bound = olGeom._geomFact.getExtent();
                }
            }

            if (date1 !== "" && date2 !== "") {
                setUrl(date1, date2, 'url', wktRepresentation);
                localStorage.setItem('date1', date1);
                localStorage.setItem('date2', date2);
                localStorage.setItem('wktRepresentation', wktRepresentation);
                sendOrder();
                this.deleteEmptyOrder();
            }
        },
        showFirstDate() {
            const sat = document.getElementById("satType").value
            getFirstImage(document.getElementById("startDatepicker").value, sat)
        },
        showSecondDate() {
            const sat = document.getElementById("satType").value
            getSecondImage(document.getElementById("finishDatepicker").value, sat)
        },
        showImages() {
            var lst = [];
            for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
                if (map.getLayers().array_[i].values_['zIndex'] !== 2) {
                    lst.push(map.getLayers().array_[i])
                }
            }
            map.setLayers(lst);

            const date1 = document.getElementById("startDatepicker").value
            const date2 = document.getElementById("finishDatepicker").value
            if (date1.length > 0 && date2.length > 0) {
                getFirstImage(date1, date2)
            }
            
            document.getElementById('clearBtnI').disabled = false;
            document.getElementById('userImg').classList.remove('text-secondary');
            document.getElementById('userImg').classList.add('text-primary');

            document.getElementById('exportBtn').disabled = true;
            document.getElementById('userShp').classList.remove('text-primary');
            document.getElementById('userShp').classList.add('text-secondary');
        },
        show1() {
            if (this.show_1 === true) {
                document.getElementById('shapeType').value = 'None'
                map.removeInteraction(draw);
                map.removeInteraction(snap);
            }
            this.show_1 = !this.show_1
        },
        show2() {
            this.show_2 = !this.show_2
        },
        show3() {
            this.show_3 = !this.show_3
            this.col1 = ""
            this.col2 = ""
        },
        selectArea() {
            map.removeInteraction(draw);
            map.removeInteraction(snap);
            let val = document.getElementById('shapeType').value;
            if (val !== 'None') {
                let geometryFunction;
                if (val === 'Box') {
                    val = 'Circle';
                    geometryFunction = createBox();
                }
                draw = new Draw({
                    source: source,
                    type: val,
                    geometryFunction: geometryFunction,
                });
                draw.on('drawstart', function (evt) {
                    source.clear();
                    if (val !== 'Circle'){
                        document.getElementById('undo').disabled = false;
                        document.getElementById('abort').disabled = false;
                    }
                }, this);
                draw.on('drawend', function (evt) {
                    document.getElementById('clear').disabled = false;

                    document.getElementById('exportBtn').disabled = false;
                    document.getElementById('exportBtnL').disabled = true;
                    document.getElementById('clearBtnL').disabled = true;
                    document.getElementById('clearBtnI').disabled = true;

                    document.getElementById('undo').disabled = true;
                    document.getElementById('abort').disabled = true;

                    document.getElementById('userShp').classList.remove('text-secondary');
                    document.getElementById('userShp').classList.add('text-primary');
                }, this);
                map.addInteraction(draw);
                snap = new Snap({source: source});
                map.addInteraction(snap);
            }
        }
    }
})

Vue.component('order-row', {
    props: ['order', 'isReady'],
    template:
        '<div>' +
            '<div v-if="order.finished_at === null">' +
                '<div class="customCard" style="border: 1px solid red;">' +
                    '<div><div style="color: red">ВЫПОЛНЯЕТСЯ</div></div>' +
                    '<div>Создан: {{new Date(order.created_at).toLocaleString("ru-RU")}}</div>' +
                    '<div>Начало съемки: {{new Date(order.imagery_start).toLocaleString("ru-RU").split(",")[0]}} Окончание: {{new Date(order.imagery_end).toLocaleString("ru-RU").split(",")[0]}}</div>' +
                    '<div><button @click="deleteOrder(order)" class="btn btn-outline-danger">Удалить заказ</button></div>' +
                '</div>' +
            '</div>' +
            '<div v-if="order.finished_at !== null">' +
                '<div class="customCard" style="border: 1px solid green;">' +
                    '<div><div style="color: green">ГОТОВО</div></div>' +
                    '<div>Создан: {{new Date(order.created_at).toLocaleString("ru-RU")}}</div>' +
                    '<div v-if="order.finished_at !== null"><div>Завершен: {{new Date(order.finished_at).toLocaleString("ru-RU")}}</div></div>' +
                    '<div>Начало съемки: {{new Date(order.imagery_start).toLocaleString("ru-RU").split(",")[0]}} Окончание: {{new Date(order.imagery_end).toLocaleString("ru-RU").split(",")[0]}}</div>' +
                    '<div><button  v-if="order.finished_at !== null" @click="showResult(order.predict)" class="btn btn-primary m-1">Результат</button></div>' +
                    '<div><button  v-if="order.finished_at !== null" @click="showImage(order.poly_wkt, order.imagery_start, order.imagery_end)" class="btn btn-outline-primary m-1">Снимок</button></div>' +
                    '<div><button @click="deleteOrder(order)" class="btn btn-outline-danger">Удалить заказ</button></div>' +
                '</div>' +
            '</div>' +
        '</div>',
    methods: {
        deleteOrder(order) {
            const url_ = `${server_url}/del-order/`
            const auth = `${localStorage.getItem("Token")}\torder-id ${order.id}`

            fetch(url_, {
                method: "DELETE",
                headers: {"Accept": 'application/json', "Content-type": 'application/json', "Authorization": auth}
            }).then(response => {
                if (!response.ok) {
                    console.log(response);
                }
            })

            var index = app.createdOrders.indexOf(order);
            if (index !== -1) {
                app.createdOrders.splice(index, 1);
            }

            var index2 = app.finishedOrders.indexOf(order);
            if (index2 !== -1) {
                app.finishedOrders.splice(index2, 1);
            }
        },
        hideImage() {
            var lst = [];
            for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
                if (map.getLayers().array_[i].values_['zIndex'] !== 2) {
                    lst.push(map.getLayers().array_[i])
                }
            }

            map.setLayers(lst)
        },
        showImage(poly_wkt, imagery_start, imagery_end) {
            const poly_crs = "EPSG:4326";  // "EPSG:3857"
            console.log("PIC PIC PIC PIC");
            console.log(poly_wkt);

            var lst = [];
            for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
                if (map.getLayers().array_[i].values_['zIndex'] !== 0) {
                    lst.push(map.getLayers().array_[i])
                }
            }
            map.setLayers(lst);

            const feature = new WKT().readFeature(poly_wkt, {
              dataProjection: poly_crs,
              featureProjection: viewProjSelect,
            });
            const new_wkt = new WKT().writeGeometry(feature.getGeometry());

            const basic = "SERVICE=WMS&REQUEST=GetMap&SHOWLOGO=false&VERSION=1.3.0&MAXCC=1&WIDTH=256&HEIGHT=256&FORMAT=image/jpeg"
            const vary = `CRS=${viewProjSelect}&LAYERS=${layer}&TIME=${imagery_start}/${imagery_end}&GEOMETRY=${new_wkt}`
            var imageLayer = new ImageLayer({
                source: new ImageStatic({
                    url: `http://services.sentinel-hub.com/ogc/wms/${sent}?${vary}&${basic}`,
                    imageExtent: feature.getGeometry().getExtent() // east, north, west, south
                }),
                zIndex: 2
            });
            map.addLayer(imageLayer);
            source.clear();

            document.getElementById('clearBtnI').disabled = false;
            document.getElementById('userImg').classList.remove('text-secondary');
            document.getElementById('userImg').classList.add('text-primary');

            document.getElementById('exportBtn').disabled = true;
            document.getElementById('userShp').classList.remove('text-primary');
            document.getElementById('userShp').classList.add('text-secondary');
        },
        showResult(res) {
            clearLayers();
            document.getElementById('exportBtnL').disabled = false;
            document.getElementById('userResG').classList.add('text-primary');
            document.getElementById('userResG').classList.remove('text-secondary');

            document.getElementById('clearBtnL').disabled = false;
            document.getElementById('userResD').classList.add('text-primary');
            document.getElementById('userResD').classList.remove('text-secondary');

            var lst_keep = [];
            for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
                if (map.getLayers().array_[i].values_['zIndex'] === 3) {
                    lst_keep.push(map.getLayers().array_[i])
                }
            }

            map.setLayers([styles[styleSelector.value]]);
            for (let i = 0, ii = lst_keep.length; i < ii; ++i) {
                map.addLayer(lst_keep[i])
            }

            var vectorSource;
            var colour = [
                'rgba(0, 0, 255, 0.3)',
                'rgba(255, 255, 0, 0.3)',
                'rgba(255, 128, 0, 0.3)',
                'rgba(255, 0, 0, 0.3)',
                'rgba(0, 128, 0, 0.3)',
            ];
            for (let i = 0, ii = res.split('\n').length; i < ii; ++i) {
                if (res.split('\n')[i].length < 10){
                    continue;
                }

                var coordinates = JSON.parse(res.split('\n')[i]).features[0].geometry.coordinates;
                var feature;
                if ((res.split('\n')[i]).includes('MultiPolygon')){
                    feature = new Feature({
                        geometry: new MultiPolygon(coordinates)
                    });
                }
                else{
                    feature = new Feature({
                        geometry: new Polygon(coordinates)
                    });
                }
                vectorSource = new VectorSource({
                    features: [feature],
                });
                map.addLayer(
                                new VectorLayer({
                                    source: vectorSource,
                                    zIndex: 3,
                                    style: {
                                            'fill-color': colour[i],
                                            'stroke-color': 'rgba(0, 0, 0, 0)',
                                            'stroke-width': 0,
                                            'circle-radius': 5,
                                            'circle-fill-color': '#666666',
                                        },
                                })
                            );
            }
            map.getView().fit(vectorSource.getExtent());
        }
    }
});

Vue.component('orders-list', {
    props: ['orders', 'fOrders', 'cOrders'],
    data: function () {
        return {
            order: null
        }
    },
    template: '<div>' +
        '<order-card-row :order="orders"/>' +
        '<p></p>' +
        '<order-row v-for="order in cOrders" :key="order.id" :order="order" :isReady="false"/>' +
        '<p></p>' +
        '<order-row v-for="order in fOrders" :key="order.id" :order="order" :isReady="true"/>' +
        '</div>',
    created: function () {
        getOrders(this.cOrders, this.fOrders)
    }
});

var app = new Vue({
    el: '#app',
    template: '<orders-list :orders="orders" :fOrders="finishedOrders" :cOrders="createdOrders"/>',
    data: {
        msg: localStorage.getItem("Token"),
        orders: [],
        createdOrders: [],
        finishedOrders: []
    },
    created: function () {
        if (localStorage.getItem("Token") == null || localStorage.getItem("Token") == 'Bearer undefined') {
            showLoginPage()
        } else {
            if (JSON.parse(atob(localStorage.getItem("Token").split('.')[1]))["exp"] < Date.now() / 1000) {
                console.log("JWT expired")
                showLoginPage()
            } else {
                showMainPage()
            }
        }
    },
    watch: {
        msg(newValue, oldValue) {
            console.log("WATCH", newValue, oldValue)
        }
    }
})


// var ws = new WebSocket("ws://127.0.0.2:65432");
// ws.onmessage = function (evt) {
//     console.log(evt);
// }
