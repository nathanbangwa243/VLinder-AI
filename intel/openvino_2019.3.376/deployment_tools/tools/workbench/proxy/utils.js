const httpProxy = require('http-proxy');

const constants = require('./constants');

const proxy = httpProxy.createProxyServer({
    target: constants.pythonTargetHost,
    ws: true,
});

function hasSessionIDHeader(req) {
    return req.get('SESSION-ID') !== undefined;
}

function checkSessionID(id) {
    return constants.sessions.indexOf(id) !== -1;
}


function getSessionIDFromURL(url) {
    const token = 'sessionID=';
    const start = url.indexOf(token) + token.length;
    const end = url.indexOf('&');
    return url.slice(start, end)
}

function generateName() {
    const possible_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const symbols = Array(15).fill().map(() => {
        return possible_characters.charAt(Math.floor(Math.random() * possible_characters.length))
    });

    return symbols.join('');
}

function isTLSSecure() {
    return !constants.sslNoVerify;
}

function isProxyTLS() {
    return Boolean(constants.proxyKey) && Boolean(constants.proxyCert);
}

function getProxyProtocol() {
    return isProxyTLS() ? 'https:' : 'http:';
}

function getWebSocketProtocol() {
    return constants.apiTLS ? 'wss:' : 'ws:';
}

module.exports = {
    hasSessionIDHeader, checkSessionID, getSessionIDFromURL, proxy, generateName,
    isProxyTLS, isTLSSecure, getProxyProtocol, getWebSocketProtocol
};