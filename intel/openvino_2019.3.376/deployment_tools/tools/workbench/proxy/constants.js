const path = require('path');


if (Boolean(process.env.PROXY_KEY) ^ Boolean(process.env.PROXY_CERT)) {
    if (process.env.PROXY_KEY) {
        throw new Error("PROXY_KEY environment variable is undefined");
    } else {
        throw new Error("PROXY_CERT environment variable is undefined");
    }
}


const sslNoVerify = Boolean(process.env.SSL_NO_VERIFY);
const apiHost = process.env.API_HOST_ADDRESS || '127.0.0.1';
const apiPort = process.env.API_PORT || 5676;
const apiTLS = Boolean(process.env.API_TLS);
const proxyHost = process.env.PROXY_HOST_ADDRESS || '127.0.0.1';
const proxyPort = process.env.PROXY_PORT || 5675;
const pythonTargetHost = `${apiTLS ? 'https:' : 'http:'}//${apiHost}:${apiPort}`;
const wsTargetHost = `${apiTLS ? 'wss:' : 'ws:'}//${proxyHost}:${proxyPort}`;
const rootPath = path.join(__dirname, '..');
const dataPath = path.join(rootPath, 'app', 'data');
const staticPath = process.env.STATIC_PATH || path.join(rootPath, 'static');
const indexHtmlPath = path.join(staticPath, 'index.html');
const isDevelopmentMode = Boolean(process.env.DEVELOPMENT_MODE);
const proxyKey = process.env.PROXY_KEY;
const proxyCert = process.env.PROXY_CERT;
const proxyKeyPassphrase = process.env.PROXY_KEY_PASSPHRASE;
const sessions = Array();
const modelsDirectory = 'models';
const originalModelDirectory = 'original';
const defaultModelArchiveExtension = '.tar.gz';


module.exports = {
    sslNoVerify,
    proxyKey,
    proxyCert,
    proxyKeyPassphrase,
    pythonTargetHost,
    wsTargetHost,
    staticPath,
    indexHtmlPath,
    proxyHost,
    proxyPort,
    isDevelopmentMode,
    dataPath,
    sessions,
    modelsDirectory,
    originalModelDirectory,
    defaultModelArchiveExtension
};
