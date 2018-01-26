require 'net/http'
require 'json'

resp = Net::HTTP::post(URI('http://localhost:7878/simulation'), File.read('../config/example.json'), 'Content-Type' => 'application/json')
p JSON.parse(resp.body)
