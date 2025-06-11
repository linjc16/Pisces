function update_table(sid, what, x)
{
    request = new XMLHttpRequest();
    addr ='/update/' + sid + '/' + what + '/' + x + '/'
    var inputs = document.getElementsByClassName('ase-input');
    sep = '?'
    for (var i = 0; i < inputs.length; i++) {
        addr += sep + inputs[i].name + '=' + inputs[i].value;
        sep = '&';
    }
    request.open('GET', addr, true);
    request.onloadstart = function() {
        table = document.getElementById('database1')
        table.innerHTML = '<p class="text-center">Searching...</p>';
    }
    request.onload = function() {
        data = request.responseText;
        table = document.getElementById('database1')
        table.innerHTML = data;
    }
    request.send();
}
