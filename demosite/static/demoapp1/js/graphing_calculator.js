// Example starter JavaScript for disabling form submissions if there are invalid fields and adding spinner
(function() {
  'use strict';
  window.addEventListener('load', function() {
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    var forms = document.getElementsByClassName('needs-validation');
    // Loop over them and prevent submission
    var validation = Array.prototype.filter.call(forms, function(form) {
      form.addEventListener('submit', function(event) {
        if (form.checkValidity() === false) {
          event.preventDefault();
          event.stopPropagation();
        }
        form.classList.add('was-validated');

        // spinner
        $('#form').on('change', function() {
          $(".btn").click(function() {
            // disable button
            // $(this).prop("disabled", true);
            // add spinner to button
            $('#spinner').html(
              `<div class="d-flex align-items-center">
                <strong>Loading...</strong>
                <div class="spinner-border ml-auto" role="status" aria-hidden="true"></div>
              </div>`
            );
          });
        });

      }, false);

    });

  }, false);

})();
