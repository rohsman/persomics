fileHandlerApp.controller('fileUpploadCtrl', ['$rootScope', '$scope', 'Upload', '$element', function($rootScope, $scope, Upload, $element) {
    $scope.uploadStatus = {
        status: 'ready',
        progress: '0%'
    };

    $scope.uploadFile = function(event) {
        $scope.uploadStatus.status = 'ongoing';
        
        Upload.upload({
            url: '/upload', 
            data: {
                transferId: Date.now(),
                imageFile: $scope.imageFile,    
                annotationFile: $scope.annotationFile,
                email: document.getElementById("email").value // Workaround since autofillable directive destroys ng-model, $scope.email
            } 
        }).then(function (resp) {
            if (resp.data.error_code === 0) {
                uploadSuccesful(resp);
            } else {
                uploadError(resp);
            }
        }, 
        function (resp) { //catch error
            uploadError(resp);
        }, 
        function (evt) {
            var progressPercentage = parseInt(100.0 * evt.loaded / evt.total);
            $scope.uploadStatus.progress = progressPercentage + '% ';
        });
    }

    function uploadSuccesful(response) {
        $scope.uploadStatus.status = 'ready';
        $rootScope.$broadcast('uploadComplete', response.config.data.imageFile.name, response.config.data.transferId);
    }

    function uploadError(response) {
        $scope.uploadStatus.status = 'aborted';
        $scope.$apply();

        if (response.data && response.data.error_desc) {
            alert('Upload unsuccessful due to: ' + response.data.error_desc);
        } else {
            alert('Upload unsuccessful due to network connection error');                        
        }
    }
}]);