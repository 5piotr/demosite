// file size validation, path writing, spinner
$('#validatedCustomFile').on('change', function() {

    const size =
       (this.files[0].size / 1024 / 1024).toFixed(2);

    if (size > 4) {
        alert("File must be smaller than 4 MB");
        $(this).val('');
    } else {
      //get the file name
      var fileName = $(this).val().replace('C:\\fakepath\\', " ");
      //replace the "Choose a file" label
      $(this).next('.custom-file-label').html(fileName);
      // spinner
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
    }
});
