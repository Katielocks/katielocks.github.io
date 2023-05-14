$.ajax({
    url: 'example.py',
    type: 'POST',
    data: JSON.stringify({x: 2, y: 3}),
    contentType: 'application/json',
    dataType: 'json',
    success: function(data) {
        console.log(data.result); // Output: 5
    },
    error: function(jqXHR, textStatus, errorThrown) {
        console.log('Error:', errorThrown);
    }
});