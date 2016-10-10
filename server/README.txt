********************
  SETUP FOLDER APP
********************

1) Setup on server:
- OSX:
    > ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
- Linux: 
    > ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"
    Follow instructions to add paths
    Exit and re-enter terminal
> brew install zeromq
> brew install pkg-config
> npm install

2) Configure port and email settings if needed in the config file:
config.js

3) Follow instructions in file persomics_image_analysis.py to setup python script.
Run python script on server:
> python persomics_image_analysis.py 0 0

4) Run node application on server: 
> node app.js

5) Open client in browser:
http://hostname:3000
- hostname = server host name, that is 'localhost' if the server is run locally
- replace 3000 with the port number configured in config.js


API
----------------
Show uploading interface: /
List processed files for specific transfer: /#?id=[transferId]
Remove all previous uploaded and processed files: /clean