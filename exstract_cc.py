#Using "whois" command, exstract country code and the destination place from the result.

import commands

file = open('log.txt','r')
ips = []

for line in file:
    ips.append(line.split('\n')[0])

print len(ips)
new_cc = []
jp=0
us=0
ru=0
for ip in ips:
    res = ''
    cc = ''
    des = ''
    command = 'whois '+ip
    res = commands.getstatusoutput(command)[1]
    w = res.split('\n')

    for i in xrange(len(w)):
        
        if 'country:' in w[i] and 'remarks' not in w[i]:
            if 'country:     ' in w[i]:
                cc = str(w[i]).split('country:     ')[1]

            else:
                cc = str(w[i]).split('country:        ')[1]
        
        if 'descr:' in w[i]:
            des = str(w[i]).split('descr:        ')[1]

        elif cc=='' or des=='':
            if 'Country:'  in w[i]:
                cc = str(w[i]).split('Country:        ')[1]
        
            if 'OrgName:' in w[i]:
                des = str(w[i]).split('OrgName:        ')[1]

    if 'JP' in cc:
        jp = jp+1
    elif 'US' in cc:
        us = us+1
    elif 'RU' in cc:
        ru = ru+1
    else:
        new_cc.append(cc)
        print 'NEW: '+ ip+' '+cc+' '+ des

print 'Total is JP: %d' % jp,' US: %d ' % us,' RU: %d'%ru, 'NEW: %s'%new_cc
