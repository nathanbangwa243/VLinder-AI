const express = require('express');
const path = require('path');
const fs = require('fs');

const constants = require('./constants');
const utils = require('./utils');

const router = express.Router();

router.get('/login', (req, res) => {
    let sessionID = '';

    do {
        sessionID = utils.generateName();
    } while (constants.sessions.indexOf(sessionID) !== -1);

    console.log(`New session: ${sessionID}`);
    constants.sessions.push(sessionID);
    res.send({'sessionID': sessionID})
});

router.get('/download/*', (req, res) => {
    console.log('Requested: ' + req.query.path);
    const archivePath = fs.realpathSync(req.query.path);
    console.log('Real: ' + archivePath);
    if (!archivePath.startsWith(fs.realpathSync(process.env.OPENVINO_WORKBENCH_DATA_PATH))) {
      res.status(404).send('Not found.');
      return;
    }
    fs.access(archivePath, fs.F_OK, (err) => {
        if (err && err.errno === 34) {
            res.status(404)
                .send('An archive with a model was not found');
            return;
        } else if (err) {
            console.log(err);
            res.status(500)
                .send('An archive with a model is corrupted');
            return;
        }

        const src = fs.createReadStream(archivePath);
        src.pipe(res);
    });
});

function directoryExists(directory) {
    return new Promise((resolve) => {
        fs.stat(directory, function (err) {
            resolve(err);
        });
    });
}

module.exports = router;
