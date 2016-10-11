var config = module.exports = {};

// Node server settings
// Use port 80 to be able to write urls without port number in a browser
config.server_port = 8080;

// Remove uploaded files after image is processed
config.clean_temporary_files = true;

// Server directories
config.directories = {
	download: '/home/karin/server/downloads',
	upload: '/home/karin/server/uploads',
	processing: '/home/karin/server/processing',
	zip: '/home/ubuntu/karin/downloadzips'
};

// Python image analysis algorithm settings
// Must be the same as configured in the python code
config.python = {
	port: 3000,
	ip: '172.31.31.65',
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
