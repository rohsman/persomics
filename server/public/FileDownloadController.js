fileHandlerApp.controller('fileDownloadCtrl', ['$scope', '$location', '$http', '$timeout', function($scope, $location, $http, $timeout) {
    $scope.downloadStatus = {
        isProcessing: false,
        text: 'No files ready for download yet'
    };

    $scope.downloadUrl = '';
    $scope.downloadAllFilesZipUrl = '';
    $scope.files = [];

    $scope.$on('uploadComplete', function(event, filename, transferId) {
        $location.search('id', transferId);
        $scope.downloadStatus.text = 'Processing ' + filename + '...';
        $scope.downloadStatus.isProcessing = true;
        $scope.files = [];
    });

    function getFilesToDownload(transferId, text) {
        $scope.downloadStatus.isProcessing = true;
        if (!text) {
            $scope.downloadStatus.text = 'Listing files...';
        }

        $http.get('result', {params: {transferId: transferId}})
        .then(
            function success(response) {
                // Use timeout service to avoid $apply to synchronous calls
                $timeout(function () {
                    if (response.data.files.length > 0) {
                        $scope.downloadUrl = transferId + '/';
                        $scope.downloadAllFilesZipUrl = transferId + '.zip';
                        $scope.files = response.data.files;
                        $scope.downloadStatus.text = text;
                    } else {
                        $scope.downloadStatus.text = 'No processed files found';
                    }
                    $scope.downloadStatus.isProcessing = false;
                }, 0);
            },
            function error(response) {
                console.error(response);
                // Use timeout service to avoid $apply to synchronous calls
                $timeout(function () {
                    $scope.downloadStatus.text = 'No processed files found';
                    $scope.downloadStatus.isProcessing = false;
                }, 0);
            }
        );
    }

    // handles events from server   
    var handleFilesReadyForDownload = function(event) {
        $scope.$apply(function () {
            var msg = JSON.parse(event.data);
            if (parseInt(msg.transferId) !== parseInt($location.search().id)) {
                // Invalid transfer, ignore
                console.log("Invalid");
                return;
            } else if (msg.error) {
                $scope.downloadStatus.text = 'Processing error: ' + msg.error;
                $scope.downloadStatus.isProcessing = false;

            } else {
                $scope.downloadStatus.text =  'Zipping result...';   
                $scope.downloadUrl = msg.transferId + '/';
                getFilesToDownload(msg.transferId, 'File processed: ' + msg.image);

            }
        });
    }

    var handleConnectionOpen = function(event) {
        if ($location.search().id) {
            getFilesToDownload($location.search().id);
        }
    }

    var handleConnectionError = function(event) {
        $scope.$apply(function () {
            console.log(event);
            if ($scope.downloadStatus.isProcessing) {
                $scope.downloadStatus.isProcessing = false;
                $scope.downloadStatus.text = 'Connection to server lost. Trying to reconnect...';
            } // else ignore
        });
    }

    var source = new EventSource('/download');
    source.addEventListener('message', handleFilesReadyForDownload, false);
    source.addEventListener('open', handleConnectionOpen, false);
    source.addEventListener('error', handleConnectionError, false);
}]); 