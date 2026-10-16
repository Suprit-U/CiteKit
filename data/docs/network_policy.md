# Campus Network and Wi-Fi

## Guest network

The guest wireless network `campus-guest` requires a portal login and disconnects
automatically after twelve hours. One device may be connected at a time. The
portal is available at `wifi.example.edu/guest`; the username is the student
registration number and the password is the last four digits of the registration
number followed by the year of joining, for example `12342024`.

## Student network

The secured network `campus-secure` uses WPA3-Enterprise with per-user credentials
issued by the network administrator. Connectivity is provided at up to 300 Mbps
on wired ports in the laboratory blocks. Residential floors are limited to 100 Mbps
per device through the access point.

## DNS

The campus resolver is `10.0.0.53` for both IPv4 and IPv6. Hosts that hard-code
public resolvers lose access to internal services, so leave DNS on automatic.

## Bandwidth limits

Each student account is capped at 30 GB per month for the secured network and
2 GB per day on the guest network. When the cap is reached the connection is
throttled rather than blocked, and the portal shows remaining quota. Exceeding
the cap three times in a semester leads to suspension of network access until the
next semester.

## Guest certificate

Guest devices must accept the `campus-ca` root certificate. Devices that do not
are redirected to a warning page on the portal. Report a certificate warning to
the IT help desk at `helpdesk@example.edu`.