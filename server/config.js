var config = module.exports = {};

// Node server settings
// Use port 80 to be able to write urls without port number in a browser
config.server_port = 3000;

// Remove uploaded files after image is processed
config.clean_temporary_files = true;

// Server directories
config.directories = {
	download: 'downloads',
	upload: 'uploads',
	processing: 'processing',
	zip: 'downloadzips'
};

// Python image analysis algorithm settings
// Must be the same as configured in the python code
config.python = {
	port: 4242,
	ip: '127.0.0.1',
	timeout: 600 // the number of seconds the node server will wait for a response before considering the analysis algorithm timed out
};

// Send email notification when image analysis is finished, 
// if email is configured
config.email = {
	user: 'folder@persomics.com',
	password: 'Xhroma1999@101',
	host: 'smtp.persomics.com',
	port: '587'
};