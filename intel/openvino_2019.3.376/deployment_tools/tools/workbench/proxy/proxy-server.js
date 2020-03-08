const cors = require('cors');
const express = require('express');
const helmet = require('helmet');
const fs = require('fs');

const utils = require('./utils');
const allRoutes = require('./routing.js');
const constants = require('./constants');

const protocol = utils.isProxyTLS() ? require('https') : require('http');

const proxyApp = express();
proxyApp.use(helmet());
proxyApp.use(helmet.frameguard({ action: 'deny' }));
proxyApp.use(helmet.permittedCrossDomainPolicies());
proxyApp.use(helmet.contentSecurityPolicy({
    directives: {
        defaultSrc: ['\'self\''],
        scriptSrc:
            constants.isDevelopmentMode ?
                ['\'self\'', '\'unsafe-eval\''] :
                ['\'self\''],
        styleSrc: ['\'self\'', '\'unsafe-inline\''],
        fontSrc: ['\'self\'', 'data:'],
        connectSrc: ['\'self\'', constants.wsTargetHost],
        frameAncestors: ['\'none\''],
    }
}));

const proxyServer = protocol.createServer(utils.isProxyTLS() ? {
        key: fs.readFileSync(constants.proxyKey),
        cert: fs.readFileSync(constants.proxyCert),
        passphrase: constants.proxyKeyPassphrase
    } : {},
    proxyApp);

const proxy = utils.proxy;

proxyApp.use(cors());

proxyApp.use(express.static(
    constants.staticPath,
    {
        index: false
    }));
    
proxyApp.use('/api', allRoutes);

// Listen for the API calls
proxyApp.all('/api/*', (req, res) => {
    if (!utils.hasSessionIDHeader(req) && !utils.checkSessionID(req.get('SESSION-ID'))) {
        res.sendStatus(503);
        return;
    }
    utils.proxy.web(req, res, {
        target: constants.pythonTargetHost,
        secure: utils.isTLSSecure()
    });
});

// Listen for not-API calls which is the Angular routing
// let the Angular decide what to do with that route
proxyApp.get('/*', (req, res) => {
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    fs.createReadStream(constants.indexHtmlPath).pipe(res);
});

// proxy the socket.io WS requests
proxyServer.on('upgrade', (req, socket, head) => {
    const sessionID = utils.getSessionIDFromURL(req.url);
    if (! utils.checkSessionID(sessionID)){
        return;
    }
    console.log('---------- SOCKET CONNECTION UPGRADING ----------');
    proxy.ws(req, socket, head, {secure: utils.isTLSSecure()});
});

utils.proxy.on('error', (err) => {
   if (err && err.code === 'ECONNREFUSED'){
       console.log('Backend server is unavailable. Waiting for the connection');
   } else {
       console.log(err);
   }
});

proxyServer.listen(constants.proxyPort, constants.proxyHost, () => {
    console.log(`Proxy server listening on ${utils.getProxyProtocol()}//${constants.proxyHost}:${constants.proxyPort}`);
});
