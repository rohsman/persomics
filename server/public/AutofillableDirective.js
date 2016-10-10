fileHandlerApp.directive('autofillable', ['$timeout', function ($timeout) {
    return {
        scope: true,
        require: 'ngModel',
        link: function (scope, elem, attrs, ngModel) {
            scope.check = function(){
                var val = elem[0].value;
                if(ngModel.$viewValue !== val){
                    ngModel.$setViewValue(val);
                    ngModel.$render();
                    elem[0].ngModel = val;
                }
                $timeout(scope.check, 300);
            };
            scope.check();
        }
    }
}]);