 /**
  * Setup on server:
  * - OSX:
  *     > ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
  * - Linux: 
  *     > ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"
  *     Follow instructions to add paths
  *     Exit and re-enter terminal
  * > brew install zeromq
  * > brew install pkg-config
  * > npm install
  *
  * Configure port and email settings if needed in the config file:
  * config.js
  *
  * Run node application on server: 
  * > node app.js
  *
  * Open client in browser:
  * http://localhost:3000
  * (replace localhost with server address if the server is not local)
  *
  *
  * Example on how to remove port from url (when another webserver is using port 80):
  * Setup proxy pass in vhosts - 
    <VirtualHost *:80>
        DocumentRoot "/Library/WebServer/Documents/www.mysite.com"
        ServerName local.www.mysite.com
        ServerAlias local.www.mysite.com
        ProxyPass /src !
        ProxyPass / http://local.www.mysite.com:3000/
        ProxyPassReverse / http://local.www.mysite.com:3000/
    </VirtualHost>
  *
  **/

    require('console-stamp')(console);

    var LONG_TIMEOUT_MS = 1000*60*60*3;

    var config = require('./config'); 
    var express = require('express'); 
    var app = express(); 
    var bodyParser = require('body-parser');
    var multer = require('multer');
    var http = require('http');
    var zerorpc = require("zerorpc");
    var fs = require('fs');
    var path = require('path');
    var nodemailer = require('nodemailer');
    var rmdir = require('rmdir');
    var os = require("os");
    var zipFolder = require('zip-folder');
    
    app.use(function(req, res, next) { //allow cross origin requests
        res.setHeader('Access-Control-Allow-Methods', 'POST, PUT, OPTIONS, DELETE, GET');
        res.header("Access-Control-Allow-Origin", "http://localhost");
        res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
        next();
    });

    // Serving from the same express Server. No cors required.
    app.use(express.static(config.directories.public));
    app.use(express.static(config.directories.download));
    app.use(express.static(config.directories.zip));
    app.use(bodyParser.json());  

    // List of connected clients, listening for processing events
    var openConnections = [];

    var storage = multer.diskStorage({ //multers disk storage settings
        destination: function (req, file, cb) {
            var transferDirectory = config.directories.upload + '/' + req.body.transferId;
            if (!fs.existsSync(transferDirectory)) {
                fs.mkdirSync(transferDirectory);
            }

            cb(null, transferDirectory);
        },
        filename: function (req, file, cb) {
            cb(null, file.originalname);
        }
    });

    /*var upload = multer({ //multer settings
                    storage: storage
                }).single('file');*/

    var upload = multer({ //multer settings
                    storage: storage
                }).fields([
                    { name: 'imageFile', maxCount: 1 }
                ]);

    var pythonClient = new zerorpc.Client({ timeout: config.python.timeout, heartbeatInterval: LONG_TIMEOUT_MS });
    pythonClient.connect('tcp://' + config.python.ip + ':' + config.python.port);

    /** API path that will upload the files */
    app.post('/upload', function(req, res) {
        console.log('Uploading...');
        upload(req, res, function(err) {
            if(err) {
                // An error occured
                console.error(err);
                res.json({error_code:1,err_desc:err});
                return;
            }
            // Successful upload
            console.log('Upload complete');
            res.json({error_code:0,err_desc:null});

            processImage(req.files['imageFile'][0].filename, req.body.annotationFile, req.body.email, req.body.transferId);
        });
    });

    /** API path that will remove all uploaded and processed files */
    app.get('/clean', function(req, res) {
        console.log('Cleaning server...');
        Object.keys(config.directories).forEach(function(key) {
            var directory = config.directories[key];
            var files = fs.readdirSync(directory);
            files.forEach(function remove(file) {
                console.log(directory + '/' + file);
                rmdir(directory + '/' + file, function(err, dirs, files) {});
            });
        });

        console.log('Cleaning complete');
        res.send('All uploaded and processed files are removed.');
    });

    /** API path that will list already uploaded files */
    app.get('/result', function(req, res) {
        console.log('Listing result for transfer ' + req.query.transferId);

        var files = getFilesReadyForDownload(req.query.transferId); 

        if (files.length > 0) {
            // Zip all files in the download directory to enable simplified download
            zipDownloads(req.query.transferId, function() { res.send({ files: files }); });
        } else {
            res.send({ files: files });
        }
    });

    function zipDownloads(transferId, callback) {
        if (!fs.existsSync(config.directories.zip)) {
            fs.mkdirSync(config.directories.zip);
        } else if (fs.existsSync(config.directories.zip + '/' + transferId + '.zip')) {
            // Do nothing if zip file already exists
            callback();
            return;
        }

        console.log('Zipping files for transfer ' + transferId);    
        zipFolder(config.directories.download + '/' + transferId + '/', config.directories.zip + '/' + transferId + '.zip', function(err) {
            if(err) {
                console.log('Zip archive could not be created', err);
            } else {
                console.log('Zip archive created');
            }
            callback();
        });
    }

    function getFilesReadyForDownload(transferId) {
        var directory = config.directories.download + '/' + transferId;
        var files = [];
        if (fs.existsSync(directory)) {
            files = fs.readdirSync(directory); 
        }   
        // Return all visible (non-hidden) files
        return files.filter(item => !(/(^|\/)\.[^\/\.]/g).test(item));
    }

    function processImage(imageFileName, annotationFileName, email, transferId) {
        console.log('Analysing ' + imageFileName + ' with annotation file ' + annotationFileName + ' for ' + email);

        pythonClient.invoke('RPC_annotation_from_reference_step_3', transferId, annotationFileName, function(err, res, more) {
            console.log('Analysis complete');

            var msg = { 
                image: imageFileName,
                transferId: transferId
            };

            if (err) {
                // Connection failure
                console.error(err);
                msg.error = err.name;
            } else if (res === 1) {
                // Algorithm failure
                msg.error = 'Image could not be processed. Sample distribution too irregular.';
                console.error('AlgorithmError: ' + msg.error);
            }

            notifyClients(msg);

            if (config.email) {
                sendEmail(email, createEmailSubject(msg), createEmailHtmlText(msg));                
            }

            // Remove temporary files
            if (config.clean_temporary_files) {
                rmdir(config.directories.upload + '/' + transferId, function(err, dirs, files) {});
                rmdir(config.directories.processing + '/' + transferId, function(err, dirs, files) {});                
            }
        });
    }

    function notifyClients(msg) {
        console.log('Notifying connected clients');
        openConnections.forEach(function(resp) {
            var id = (new Date()).getMilliseconds();

            resp.write('data:' + JSON.stringify(msg) + '\n\n'); // Note the extra newline
        });
    }

    function createEmailSubject(msg) {
        var subject;
        if (msg.error) {
            subject = 'Persomics file processing error';
        } else {
            subject = 'Persomics file processed';
        }
        return subject;
    }

    function createEmailHtmlText(msg) {
        var htmlText;
        if (msg.error) {
            htmlText = '<p>Your file <i>' + msg.image + '</i> could not be processed.</p><p>' + msg.error + '</p>';
        } else {
            var downloadUrl = config.server.address + '/#?id=' + msg.transferId;
            htmlText = '<p>Your file <i>' + msg.image + '</i> have been processed.</p><p>Downloads: <a href="' + downloadUrl + '">' + downloadUrl + '</a></p><ul>';
        }
        return htmlText;
    }

    function sendEmail(toAddress, subject, htmlText) {
        var transporter = nodemailer.createTransport({
            host: config.email.host,
            port: config.email.port,
            auth: {
                user: config.email.user, 
                pass: config.email.password
            },
            tls: {
                rejectUnauthorized: false
            }
        });

        var mailOptions = {
            from: config.email.user, // sender address
            to: toAddress, // list of receivers
            subject: subject, // Subject line
            //text: '', // use plaintext body...
            html: htmlText // ...or HTML body instead
        };

        transporter.sendMail(mailOptions, function(error, info){
            if (error) {
                console.log('Email could not be sent to ' + toAddress + ': ' + error);
            } else {
                console.log('Email sent to ' + toAddress + ': ' + info.response);
            };
        });
    }

    /** API path that will register client for download events */
    app.get('/download', function(req, res) {
        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        });
        res.write('\n');
     
        // Push this res object to our global variable
        openConnections.push(res);
        console.log('Client connected. Total ' + openConnections.length);
     
        // When the request is closed, e.g. the browser window is closed. We search through the open connections array and remove this connection.
        req.on("close", function() {
            var toRemove;
            for (var j =0; j < openConnections.length; j++) {
                if (openConnections[j] == res) {
                    toRemove =j;
                    break;
                }
            }
            openConnections.splice(j,1);

            console.log('Client disconnected. Total ' + openConnections.length);
        });
    });

    var server = app.listen(config.server.port, function() {
        console.log('Running on port ' + config.server.port);
        console.log('Hostname: ' + os.hostname());
    });
    server.timeout = LONG_TIMEOUT_MS;
